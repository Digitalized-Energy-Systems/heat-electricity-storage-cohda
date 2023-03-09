"""Module for distributed real power planning with COHDA. Contains roles, which
integrate COHDA in the negotiation system and the core COHDA-decider together with its model.
"""
import asyncio
import logging
import time
from copy import copy
from typing import List, Dict, Any, Tuple, Optional, Callable

import numpy as np
from mango.messages.codecs import json_serializable

from mango_library.coalition.core import CoalitionAssignment
from mango_library.negotiation.cohda.data_classes import \
    SolutionCandidate, SystemConfig, WorkingMemory, ScheduleSelection, EnergySchedules
from mango_library.negotiation.core import NegotiationParticipant, NegotiationStarterRole, Negotiation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)
# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

printarray = [
    "-start_values",
    "-start _decide",
    "-open_schedule",
    "-max_value_schedules",
]


class Colors:
    RESET_ALL = "\033[0m"

    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINED = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    HIDDEN = "\033[8m"

    RESET_BOLD = "\033[21m"
    RESET_DIM = "\033[22m"
    RESET_UNDERLINED = "\033[24m"
    RESET_BLINK = "\033[25m"
    RESET_REVERSE = "\033[27m"
    RESET_HIDDEN = "\033[28m"

    DEFAULT = "\033[39m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    LIGHT_GRAY = "\033[37m"
    DARK_GRAY = "\033[90m"
    LIGHT_RED = "\033[91m"
    LIGHT_GREEN = "\033[92m"
    LIGHT_YELLOW = "\033[93m"
    LIGHT_BLUE = "\033[94m"
    LIGHT_MAGENTA = "\033[95m"
    LIGHT_CYAN = "\033[96m"
    WHITE = "\033[97m"

    BACKGROUND_DEFAULT = "\033[49m"
    BACKGROUND_BLACK = "\033[40m"
    BACKGROUND_RED = "\033[41m"
    BACKGROUND_GREEN = "\033[42m"
    BACKGROUND_YELLOW = "\033[43m"
    BACKGROUND_BLUE = "\033[44m"
    BACKGROUND_MAGENTA = "\033[45m"
    BACKGROUND_CYAN = "\033[46m"
    BACKGROUND_LIGHT_GRAY = "\033[47m"
    BACKGROUND_DARK_GRAY = "\033[100m"
    BACKGROUND_LIGHT_RED = "\033[101m"
    BACKGROUND_LIGHT_GREEN = "\033[102m"
    BACKGROUND_LIGHT_YELLOW = "\033[103m"
    BACKGROUND_LIGHT_BLUE = "\033[104m"
    BACKGROUND_LIGHT_MAGENTA = "\033[105m"
    BACKGROUND_LIGHT_CYAN = "\033[106m"
    BACKGROUND_WHITE = "\033[107m"


""" globals """
start_time = {Colors.BACKGROUND_LIGHT_MAGENTA: time.time()}
global_start_time = time.time()
best_perf = {}
best_counter = 0
best_counter_end = False
""" globals """


def get_time(color):
    global start_time
    global global_start_time
    if color not in start_time.keys():
        start_time[color] = time.time()
    time_diff = time.time() - start_time[color]
    start_time[color] = time.time()
    return [time_diff, start_time[color] - global_start_time]


def test_print(param):
    length = 0
    for val in printarray:
        if not (len(val) <= length or val[0] == "-"):
            length = len(val)
    length -= len(param)
    part = int(length / 2)
    if any([item == param for item in printarray]):
        print(f"{Colors.BOLD}{Colors.YELLOW}{' ' * part}{param}{' ' * (length - part)}{Colors.RESET_ALL}", end=" ")
        return True
    elif any([item == "-" + param for item in printarray]):
        return False
    else:
        print(f"{Colors.BOLD}{Colors.BACKGROUND_RED}{' ' * part}{param}{' ' * (length - part)}{Colors.RESET_ALL}", end=" ")
        return True


@json_serializable
class CohdaMessage:
    """
    Message for a COHDa negotiation.
    Contains the candidate and the working memory of an agent.
    """

    def __init__(self, working_memory: WorkingMemory):
        self._working_memory = working_memory

    @property
    def working_memory(self) -> WorkingMemory:
        """Return the working memory of the sender agent

        :return: the working memory of the sender
        """
        return self._working_memory


class CohdaNegotiationStarterRole(NegotiationStarterRole):
    """Convenience role for starting a COHDA negotiation with simply providing a target schedule
    """

    " create an empty Working memory and send it together with the target params "

    def __init__(self, target_params, coalition_model_matcher=None, coalition_uuid=None) -> None:
        """

        :param target_params: Parameter that are necessary for the agents to calculate the performance.
        Could be e.g. the target schedule.
        :param coalition_model_matcher:
        :param coalition_uuid:
        """
        super().__init__(
            lambda assignment:
            CohdaMessage(WorkingMemory(target_params=target_params, system_config=SystemConfig({}),
                                       solution_candidate=SolutionCandidate(
                                           agent_id=assignment.part_id,
                                           schedules={}
                                       ))),
            coalition_model_matcher=coalition_model_matcher, coalition_uuid=coalition_uuid
        )


class COHDA:
    """COHDA-decider
    """

    def __init__(self, schedule_provider, is_local_acceptable, part_id: str, value_weights, open_value_weights, perf_func=None):
        self._schedule_provider = [EnergySchedules(dict_schedules={'power': np.array(schedule), 'heat': np.array(schedule) * 0}) for schedule in schedule_provider]
        self._is_local_acceptable = is_local_acceptable
        self._memory = WorkingMemory(None, SystemConfig({}), SolutionCandidate(part_id, {}))
        self._best_solution_candidate = None
        self._counter = 0
        self._part_id = part_id
        self._value_weights = value_weights
        # TODO entfernen open_value_weights
        self._open_value_weights = open_value_weights
        if perf_func is None:
            self._perf_func = self.get_perf
        else:
            self._perf_func = perf_func

    def print_color(self, text, color=Colors.RESET_ALL):
        print(Colors.RESET_ALL, end="")
        print(Colors.BOLD, end="")
        print(color, end="")
        if color is Colors.RESET_ALL:
            print(f"agent{self._part_id}: {text}")
        else:
            time_diffs = get_time(color)
            print(f"agent{self._part_id}: {format(time_diffs[0] * 1000, '.3f')}ms - {format(time_diffs[1], '.6f')}s - {text}")
        print(Colors.RESET_ALL, end="")

    @staticmethod
    def deviation_to_target_schedule(cluster_schedule: SolutionCandidate, target_parameters):
        """
        EnergySchedules(target_parameters) - EnergySchedules(cluster_schedule) = Dict
        """
        # TODO Add the use of np.array(weights)
        target_schedule, weights = target_parameters
        target_schedule = EnergySchedules(dict_schedules=target_schedule)
        for agent_id in cluster_schedule.schedules:
            target_schedule = target_schedule - cluster_schedule.schedules[agent_id]
        # TODO Can the cluster_schedule be empty? And is that important for the calculation?
        # if cluster_schedule.size == 0:
        #     return float('-inf')
        return target_schedule.sum()

    @staticmethod
    def get_perf(energy_schedules, target_schedule) -> float:
        result = 0
        for dict_key in target_schedule.dict_schedules.keys():
            test = np.abs(np.subtract(target_schedule.dict_schedules[dict_key], energy_schedules.dict_schedules[dict_key]))
            result += np.sum(test)
        return result

    def handle_cohda_msgs(self, messages: List[CohdaMessage]) -> Optional[CohdaMessage]:
        """
        This called by the COHDARole. It takes a List of COHDA messages, executes perceive, decide, act and returns
        a CohdaMessage in case the working memory has changed and None otherwise
        :param messages: The list of received CohdaMessages
        :return: The message to be sent to the neighbors, None if no message has to be sent
        """
        old_sysconf = self._memory.system_config
        old_candidate = self._memory.solution_candidate

        " perceive "
        sysconf, candidate = self._perceive(messages)

        " decide "
        if sysconf is not old_sysconf or candidate is not old_candidate:
            sysconf, candidate = self._decide(sysconfig=sysconf, candidate=candidate)
            global best_perf
            global best_counter
            global best_counter_end
            best_counter += 1
            candidate_copy = copy(candidate)
            new_best_counter = False
            if self._part_id not in best_perf.keys() or \
                    best_perf[self._part_id].perf > candidate.perf or \
                    len(best_perf[self._part_id].schedules) < len(candidate.schedules):
                self._best_solution_candidate = candidate_copy
                best_perf[self._part_id] = candidate_copy
                best_counter = 0
                new_best_counter = True
            min_perf = float("inf")
            max_perf = 0
            for best_perf_keys in best_perf.keys():
                if min_perf > best_perf[best_perf_keys].perf:
                    min_perf = best_perf[best_perf_keys].perf
                if max_perf < best_perf[best_perf_keys].perf:
                    max_perf = best_perf[best_perf_keys].perf
            diff = max_perf - min_perf
            # TODO Optimal multiplication * 20
            maximum_agent_attempts = 3
            if best_counter > len(best_perf) * maximum_agent_attempts:
                best_counter_end = True
            if best_counter_end:
                print(Colors.BOLD, end="")
            if not (new_best_counter or best_counter_end):
                if best_counter == 1:
                    print()
                print(".", end="")
                # logging.debug(f"{self._part_id}: new:{format(candidate_copy.perf, '.3f')} - {format(min_perf, '.0f')}-{format(max_perf, '.0f')} - {format((time.time() - global_start_time), '.3f')}s - diff:{np.round(diff, 3)} - {best_counter}")
            elif new_best_counter:
                print(f"\n{self._part_id}: new:{format(candidate_copy.perf, '.3f')} - {format(min_perf, '.0f')}-{format(max_perf, '.0f')} - {format((time.time() - global_start_time), '.3f')}s - diff:{np.round(diff, 3)}", end="")
            elif best_counter_end:
                print(f"\n{self._part_id}: {format(min_perf, '.0f')}-{format(max_perf, '.0f')} - {format((time.time() - global_start_time), '.3f')}s - diff:{np.round(diff, 3)}", end="")
            print(Colors.RESET_ALL, end="")
            if best_counter_end:
                return None
            " act "
            return self._act(new_sysconfig=sysconf, new_candidate=candidate)
        else:
            return None

    def _perceive(self, messages: List[CohdaMessage]) -> Tuple[SystemConfig, SolutionCandidate]:
        """
        Updates the current knowledge
        :param messages: The List of received CohdaMessages
        :return: a tuple of SystemConfig, Candidate as a result of perceive
        """
        current_sysconfig = None
        current_candidate = None
        for message in messages:
            if self._memory.target_params is None:
                " get target parameters if not known "
                self._memory.target_params = message.working_memory.target_params

            if current_sysconfig is None:
                if self._part_id not in self._memory.system_config.schedule_choices:
                    " if you have not yet selected any schedule in the sysconfig, choose any to start with "
                    schedule_choices = self._memory.system_config.schedule_choices
                    self._counter += 1
                    schedule_choices[self._part_id] = ScheduleSelection(self._schedule_provider[0], self._counter)
                    " we need to create a new class of Systemconfig so the updates are "
                    " recognized in handle_cohda_msgs() "
                    current_sysconfig = SystemConfig(schedule_choices=schedule_choices)
                else:
                    current_sysconfig = self._memory.system_config

            if current_candidate is None:
                if self._part_id not in self._memory.solution_candidate.schedules:
                    " if you have not yet selected any schedule in the sysconfig, choose any to start with "
                    schedules = self._memory.solution_candidate.schedules
                    schedules[self._part_id] = self._schedule_provider[0]
                    " we need to create a new class of SolutionCandidate so the updates are "
                    " recognized in handle_cohda_msgs() "
                    current_candidate = SolutionCandidate(agent_id=self._part_id, schedules=schedules)
                else:
                    current_candidate = self._memory.solution_candidate

            new_sysconf = message.working_memory.system_config
            new_candidate = message.working_memory.solution_candidate

            " Merge new information into current_sysconfig and current_candidate "
            current_sysconfig = self._merge_sysconfigs(sysconfig_i=current_sysconfig, sysconfig_j=new_sysconf)
            current_candidate = self._merge_candidates(candidate_i=current_candidate,
                                                       candidate_j=new_candidate,
                                                       agent_id=self._part_id,
                                                       perf_func=self._perf_func)

        return current_sysconfig, current_candidate

    def _decide(self, sysconfig: SystemConfig, candidate: SolutionCandidate) -> Tuple[SystemConfig, SolutionCandidate]:
        """
        Check whether a better SolutionCandidate can be created based on the current state of the negotiation
        :param sysconfig: Current SystemConfig
        :param candidate: Current SolutionCandidate
        :return: Tuple of SystemConfig, SolutionCandidate. Unchanged to parameters if no new SolutionCandidate was
        found. Else it consists of the new SolutionCandidate and an updated SystemConfig
        """

        if test_print("start _decide"):
            self.print_color("start _decide", Colors.BACKGROUND_LIGHT_MAGENTA)

        " calc power_open_schedule "
        # TODO use the weights in target_params
        target_schedule = EnergySchedules(dict_schedules=self._memory.target_params[0])
        open_schedule = target_schedule
        for candidate_schedule_key in candidate.schedules.keys():
            open_schedule -= candidate.schedules[candidate_schedule_key]
        if test_print("open_schedule"):
            self.print_color(f"{Colors.BOLD}ZIEL:{Colors.RESET_ALL} {target_schedule}")
            self.print_color(f"{Colors.BOLD}open_schedule{Colors.RESET_ALL} {open_schedule}")
            self.print_color(f"{Colors.BOLD}self._open_value_weights{Colors.RESET_ALL} {self._open_value_weights}")

        possible_schedules = self._schedule_provider
        current_best_candidate = candidate
        current_best_candidate.perf = float('-inf')

        max_list_of_outputs = 10
        " calculate the best gas_amount for each timestamp "
        for energy_schedule in possible_schedules:
            if self._is_local_acceptable(energy_schedule):
                schedule_with_max_values = []
                for timestamp, _ in enumerate(list(energy_schedule.dict_schedules.values())[0]):
                    max_gas_amount = self._value_weights['max_gas_amount']
                    fixed_values = {'heat_open': np.max([open_schedule.dict_schedules['heat'][timestamp], 0]),
                                    'power_open': np.max([open_schedule.dict_schedules['power'][timestamp], 0]),
                                    'power_schedule_timestamp': energy_schedule.dict_schedules['power'][timestamp]}
                    old_gas_amount = 0
                    if candidate.schedules[self._part_id] and 'gas_amount' in candidate.schedules[self._part_id].dict_schedules.keys() and candidate.schedules[self._part_id].dict_schedules['gas_amount'][timestamp]:
                        old_gas_amount = candidate.schedules[self._part_id].dict_schedules['gas_amount'][timestamp]
                    list_of_outputs = [self.test_gas_amount(old_gas_amount, fixed_values)]
                    if max_gas_amount < 4:
                        for i in range(0, max_gas_amount):
                            list_of_outputs.append(self.test_gas_amount(i, fixed_values))
                    else:
                        list_of_outputs.append(self.test_gas_amount(int(max_gas_amount * 0), fixed_values))
                        list_of_outputs.append(self.test_gas_amount(int(max_gas_amount / 4), fixed_values))
                        list_of_outputs.append(self.test_gas_amount(int(max_gas_amount / 2), fixed_values))
                        list_of_outputs.append(self.test_gas_amount(int(max_gas_amount / 4 * 3), fixed_values))
                        list_of_outputs.append(self.test_gas_amount(int(max_gas_amount * 1), fixed_values))
                        while len(list_of_outputs) < min(max_list_of_outputs, max_gas_amount):
                            list_of_outputs.sort(key=lambda x: [x['value'], -x['gas_amount']], reverse=True)
                            less_value_objects = list(filter(lambda x: (x['gas_amount'] < list_of_outputs[0]['gas_amount']), list_of_outputs))
                            if less_value_objects:
                                less_value_object = max(less_value_objects, key=lambda item: item['value'])
                                list_of_outputs.append(self.test_gas_amount(int((list_of_outputs[0]['gas_amount'] + less_value_object['gas_amount']) / 2), fixed_values))
                            more_value_objects = list(filter(lambda x: (x['gas_amount'] > list_of_outputs[0]['gas_amount']), list_of_outputs))
                            if more_value_objects:
                                more_value_object = max(more_value_objects, key=lambda item: item['value'])
                                list_of_outputs.append(self.test_gas_amount(int((list_of_outputs[0]['gas_amount'] + more_value_object['gas_amount']) / 2), fixed_values))
                    list_of_outputs.sort(key=lambda x: [x['value'], -x['gas_amount']], reverse=True)
                    schedule_with_max_values.append(list_of_outputs[0])
                max_value_schedules = {}
                for key in schedule_with_max_values[0].keys():
                    max_value_schedule = []
                    for value in schedule_with_max_values:
                        max_value_schedule.append(value[key])
                    max_value_schedules[key] = max_value_schedule
                if test_print("max_value_schedules"):
                    self.print_color((np.sum(max_value_schedules['value']), max_value_schedules))
                new_candidate_schedules = copy(current_best_candidate.schedules)
                new_candidate_schedules[self._part_id] = EnergySchedules(
                    dict_schedules={
                        'power': max_value_schedules['end_power'],
                        'heat': max_value_schedules['end_heat'],
                        'gas_amount': max_value_schedules['gas_amount'],
                        'power_to_heat': max_value_schedules['power_to_heat'],
                        'power_to_conversion': max_value_schedules['power_to_conversion'],
                    },
                    perf=np.sum(max_value_schedules['value']),
                )
                new_candidate = SolutionCandidate(agent_id=self._part_id, schedules=new_candidate_schedules)
                if new_candidate.perf > current_best_candidate.perf:
                    current_best_candidate = new_candidate

        # TODO Ist das OPEN value überhaupt notwendig? wird es angeguckt?
        schedule_in_candidate = current_best_candidate.schedules.get(self._part_id, None)
        schedule_choice_in_sysconfig = sysconfig.schedule_choices.get(self._part_id, None)

        # if schedule_choice_in_sysconfig is None or \
        #         not np.array_equal(schedule_in_candidate, schedule_choice_in_sysconfig.energy_schedules):
        if schedule_choice_in_sysconfig is None or schedule_in_candidate is not schedule_choice_in_sysconfig.energy_schedules:
            " update Sysconfig if your schedule in the current sysconf is different to the one in the candidate "
            sysconfig.schedule_choices[self._part_id] = ScheduleSelection(
                energy_schedules=schedule_in_candidate, counter=self._counter + 1)
            " update counter "
            self._counter += 1

        current_best_candidate.perf = self._perf_func(current_best_candidate.to_energy_schedules(), target_schedule)
        if test_print("start _decide"):
            self.print_color("end _decide", Colors.BACKGROUND_LIGHT_MAGENTA)

        return sysconfig, current_best_candidate

    def test_gas_amount(self, gas_amount, fixed_values):
        gas_to_power = gas_amount * self._value_weights['gas_to_power_factor']
        gas_to_heat = gas_amount * self._value_weights['gas_to_heat_factor']
        max_power_to_power_to_heat = np.min([self._value_weights['power_to_heat_amount'],
                                             fixed_values['power_schedule_timestamp'] + gas_to_power - fixed_values['power_open']])
        max_heat_in_power_to_heat = np.max([(fixed_values['heat_open'] - gas_to_heat) / self._value_weights['power_to_heat_factor'], 0])
        power_to_heat = np.max([np.min([max_power_to_power_to_heat, max_heat_in_power_to_heat]), 0])
        power_to_conversion = np.min([np.max([fixed_values['power_schedule_timestamp'] + gas_to_power - power_to_heat - fixed_values['power_open'], 0]), self._value_weights['convert_amount']])
        end_power = fixed_values['power_schedule_timestamp'] + gas_to_power - power_to_heat - power_to_conversion
        end_heat = gas_to_heat + power_to_heat * self._value_weights['power_to_heat_factor']
        value = end_power * self._value_weights['power_kwh_price'] + end_heat * self._value_weights['heat_kwh_price'] \
                + power_to_conversion * self._value_weights['converted_price'] \
                - np.absolute(end_power - fixed_values['power_open']) * self._value_weights['power_penalty'] \
                - np.absolute(end_heat - fixed_values['heat_open']) * self._value_weights['heat_penalty'] \
                - gas_amount * self._value_weights['gas_price']

        return {'gas_amount': gas_amount, 'value': value, 'end_power': end_power, 'end_heat': end_heat, 'power_to_heat': power_to_heat, 'power_to_conversion': power_to_conversion}

    def _act(self, new_sysconfig: SystemConfig, new_candidate: SolutionCandidate) -> CohdaMessage:
        """
        Stores the new SystemConfig and SolutionCandidate in Memory and returns the COHDA message that should be sent
        :param new_sysconfig: The SystemConfig as a result from perceive and decide
        :param new_candidate: The SolutionCandidate as a result from perceive and decide
        :return: The COHDA message that should be sent
        """
        " update memory "
        self._memory.system_config = new_sysconfig
        self._memory.solution_candidate = new_candidate

        " return COHDA message "
        return CohdaMessage(working_memory=self._memory)

    @staticmethod
    def _merge_sysconfigs(sysconfig_i: SystemConfig, sysconfig_j: SystemConfig):
        """
        Merge *sysconf_i* and *sysconf_j* and return the result.

        Returns a merged systemconfig. If the sysconfig_i remains unchanged, the same instance of sysconfig_i is
        returned, otherwise a new object is created.
        """

        sysconfig_i_schedules: Dict[str, ScheduleSelection] = sysconfig_i.schedule_choices
        sysconfig_j_schedules: Dict[str, ScheduleSelection] = sysconfig_j.schedule_choices
        key_set_i = set(sysconfig_i_schedules.keys())
        key_set_j = set(sysconfig_j_schedules.keys())

        new_sysconfig: Dict[str, ScheduleSelection] = {}
        modified = False

        for i, a in enumerate(sorted(key_set_i | key_set_j)):
            " An a might be in key_set_i, key_set_j or in both! "
            if a in key_set_i and \
                    (a not in key_set_j or sysconfig_i_schedules[a].counter >= sysconfig_j_schedules[a].counter):
                " Use data of sysconfig_i "
                schedule_selection = sysconfig_i_schedules[a]
            else:
                " Use data of sysconfig_j "
                schedule_selection = sysconfig_j_schedules[a]
                modified = True

            new_sysconfig[a] = schedule_selection

        if modified:
            sysconf = SystemConfig(new_sysconfig)
        else:
            sysconf = sysconfig_i

        return sysconf

    @staticmethod
    def _merge_candidates(candidate_i: SolutionCandidate, candidate_j: SolutionCandidate,
                          agent_id: str, perf_func: Callable):
        """
        Returns a merged Candidate. If the candidate_i remains unchanged, the same instance of candidate_i is
        returned, otherwise a new object is created with agent_id as candidate.agent_id
        :param candidate_i: The first candidate
        :param candidate_j: The second candidate
        :param agent_id: The agent_id that defines who is the creator of a new candidate
        :param perf_func: The performance function
        :return:  A merged SolutionCandidate. If the candidate_i remains unchanged, the same instance of candidate_i is
        returned, otherwise a new object is created.
        """
        keyset_i = set(candidate_i.schedules.keys())
        keyset_j = set(candidate_j.schedules.keys())
        candidate = candidate_i  # Default candidate is *i*

        if keyset_i < keyset_j:
            # Use *j* if *K_i* is a true subset of *K_j*
            candidate = candidate_j
        elif keyset_i == keyset_j:
            # Compare the performance if the keyset is equal
            if candidate_j.perf > candidate_i.perf:
                # Choose *j* if it performs better
                candidate = candidate_j
            elif candidate_j.perf == candidate_i.perf:
                # If both perform equally well, order them by name
                if candidate_j.agent_id < candidate_i.agent_id:
                    candidate = candidate_j
        elif keyset_j - keyset_i:
            # If *candidate_j* shares some entries with *candidate_i*, update *candidate_i*
            new_schedules: Dict[str, np.array] = {}
            for a in sorted(keyset_i | keyset_j):
                if a in keyset_i:
                    schedule = candidate_i.schedules[a]
                else:
                    schedule = candidate_j.schedules[a]
                new_schedules[a] = schedule

            # create new SolutionCandidate
            candidate = SolutionCandidate(agent_id=agent_id, schedules=new_schedules)

        return candidate


class COHDARole(NegotiationParticipant):
    """Negotiation role for COHDA.
    """

    def __init__(self, schedules_provider, value_weights, open_value_weights, local_acceptable_func=None, check_inbox_interval: float = 0.1):
        """
        Init of COHDARole
        :param schedules_provider: Function that takes not arguments and returns a list of schedules
        :param value_weights: own value_weights for calculation
        :param open_value_weights: own open_value_weights visible for other agents
        :param local_acceptable_func: Function that takes a schedule as input and returns a boolean indicating,
        if the schedule is locally acceptable or not. Defaults to lambda x: True
        :param check_inbox_interval: Duration of buffering the cohda messages [s]
        """
        super().__init__()

        self._schedules_provider = schedules_provider
        self._value_weights = value_weights
        self._open_value_weights = open_value_weights
        if local_acceptable_func is None:
            self._is_local_acceptable = lambda x: True
        else:
            self._is_local_acceptable = local_acceptable_func
        self._cohda = {}
        self._cohda_msg_queues = {}
        self._cohda_tasks = []
        self.check_inbox_interval = check_inbox_interval

    def create_cohda(self, part_id: str):
        """
        Create an instance of COHDA.
        :param part_id: participant id
        :return: COHDA object
        """
        if test_print("start_values"):
            print(f"{part_id}: {self._schedules_provider}", end=" ")
            print(f"kwh: {float(self._value_weights['power_kwh_price'])}€", end=" ")
            print(f"convert: {self._value_weights['convert_amount']}x", end=" ")
            print(f"{float(self._value_weights['converted_price'])}€", end=" ")
            print()
        return COHDA(schedule_provider=self._schedules_provider,
                     is_local_acceptable=self._is_local_acceptable,
                     part_id=part_id,
                     value_weights=self._value_weights,
                     open_value_weights=self._open_value_weights,
                     )

    async def on_stop(self) -> None:
        """
        Will be called once the agent is shutdown
        """
        " cancel all cohda tasks "
        for task in self._cohda_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def handle(self,
               message,
               assignment: CoalitionAssignment,
               negotiation: Negotiation,
               meta: Dict[str, Any]):

        if negotiation.coalition_id in self._cohda:
            negotiation.active = True
            self._cohda_msg_queues[negotiation.coalition_id].append(message)
        else:
            self._cohda[negotiation.coalition_id] = self.create_cohda(assignment.part_id)
            self._cohda_msg_queues[negotiation.coalition_id] = [message]

            async def process_msg_queue():
                """
                Method to evaluate all incoming message of a cohda_message_queue for a certain negotiation
                """

                if len(self._cohda_msg_queues[negotiation.coalition_id]) > 0:
                    # get queue
                    cohda_message_queue, self._cohda_msg_queues[negotiation.coalition_id] = \
                        self._cohda_msg_queues[negotiation.coalition_id], []

                    message_to_send = self._cohda[negotiation.coalition_id].handle_cohda_msgs(cohda_message_queue)

                    if message_to_send is not None:
                        await self.send_to_neighbors(assignment, negotiation, message_to_send)

                    else:
                        # set the negotiation as inactive as the incoming information was known already
                        negotiation.active = False
                else:
                    # set the negotiation as inactive as no message has arrived
                    negotiation.active = False

            self._cohda_tasks.append(self.context.schedule_periodic_task(process_msg_queue,
                                                                         delay=self.check_inbox_interval))
