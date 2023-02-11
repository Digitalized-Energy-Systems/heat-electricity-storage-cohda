"""Module for distributed real power planning with COHDA. Contains roles, which
integrate COHDA in the negotiation system and the core COHDA-decider together with its model.
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from mango.messages.codecs import json_serializable

from mango_library.coalition.core import CoalitionAssignment
from mango_library.negotiation.cohda.data_classes import \
    SolutionCandidate, SystemConfig, WorkingMemory, ScheduleSelection, EnergySchedules
from mango_library.negotiation.core import NegotiationParticipant, NegotiationStarterRole, Negotiation

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

start_time = time.time()
global_start_time = time.time()
printarray = [
    "-start_values",
    "start _decide",
    "-power_open_schedule",
    "-open_schedule",
    "-"
    "Calculate the schedules",
    "-pow_to_take_over_conversion",
    "-testpow_to_take_over_conversion",
]
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 10)
pd.options.display.float_format = '{:,.0f}'.format
pd.set_option('display.colheader_justify', 'center')


def get_time():
    global start_time
    global global_start_time
    time_diff = time.time() - start_time
    # print(f" - g: {format(time.time() - global_start_time, '.6f')}s - {format(time.time() - start_time, '.6f')}s - {text} -")
    start_time = time.time()
    return [time_diff, start_time - global_start_time]


class colors:
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


def test_print(param):
    if any([item == param for item in printarray]):
        return True
    elif any([item == "-" + param for item in printarray]):
        return False
    else:
        print(f'{colors.BACKGROUND_RED}test_print("{colors.BOLD}{param}{colors.RESET_ALL}{colors.BACKGROUND_RED}"){colors.RESET_ALL}')
        return False


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

    # create an empyt Working memory and send it together with the target params
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
                                           schedules={},
                                           perf=float('-inf')))),
            coalition_model_matcher=coalition_model_matcher, coalition_uuid=coalition_uuid
        )


class COHDA:
    """COHDA-decider
    """

    def __init__(self, schedule_provider, is_local_acceptable, part_id: str, value_weights, open_value_weights, perf_func=None):
        self._schedule_provider = [EnergySchedules(dict_schedules={'power': np.array(schedule), 'heat': np.array(schedule) * 0}) for schedule in schedule_provider]
        print(part_id, self._schedule_provider)
        self._is_local_acceptable = is_local_acceptable
        self._memory = WorkingMemory(None, SystemConfig({}), SolutionCandidate(part_id, {}, float('-inf')))
        self._counter = 0
        self._part_id = part_id
        self._value_weights = value_weights
        self._open_value_weights = open_value_weights
        """ for print """
        self._last = "String for printing"
        """ for print """
        if perf_func is None:
            self._perf_func = self.deviation_to_target_schedule
        else:
            self._perf_func = perf_func

    def print_color(self, text, color=colors.RESET_ALL):
        print(colors.RESET_ALL, end="")
        print(colors.BOLD, end="")
        print(color, end="")
        if color == colors.RESET_ALL:
            print(f"agent{self._part_id}: {text}")
        else:
            time_diffs = get_time()
            print(f"agent{self._part_id}: {text} - {format(time_diffs[0] * 1000, '.3f')}ms - {format(time_diffs[1], '.6f')}s")
        print(colors.RESET_ALL, end="")

    @staticmethod
    def deviation_to_target_schedule(cluster_schedule: SolutionCandidate, target_parameters, value_weights):
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
        target_schedule_sum = target_schedule.sum()
        """ Add the penaltys to the perf """
        result = 0
        for dict_key in target_schedule_sum.keys():
            result += np.sum(target_schedule_sum[dict_key] * value_weights[dict_key + '_penalty'])
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

        # perceive
        sysconf, candidate = self._perceive(messages)

        # decide
        if sysconf is not old_sysconf or candidate is not old_candidate:
            sysconf, candidate = self._decide(sysconfig=sysconf, candidate=candidate)
            # act
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
                # get target parameters if not known
                self._memory.target_params = message.working_memory.target_params

            if current_sysconfig is None:
                if self._part_id not in self._memory.system_config.schedule_choices:
                    # if you have not yet selected any schedule in the sysconfig, choose any to start with
                    schedule_choices = self._memory.system_config.schedule_choices
                    schedule_choices[self._part_id] = ScheduleSelection(self._schedule_provider[0], self._counter + 1)
                    self._counter += 1
                    # we need to create a new class of Systemconfig so the updates are
                    # recognized in handle_cohda_msgs()
                    current_sysconfig = SystemConfig(schedule_choices=schedule_choices)
                else:
                    current_sysconfig = self._memory.system_config

            if current_candidate is None:
                if self._part_id not in self._memory.solution_candidate.schedules:
                    # if you have not yet selected any schedule in the sysconfig, choose any to start with
                    schedules = self._memory.solution_candidate.schedules
                    schedules[self._part_id] = self._schedule_provider[0]
                    # we need to create a new class of SolutionCandidate so the updates are
                    # recognized in handle_cohda_msgs()
                    current_candidate = SolutionCandidate(agent_id=self._part_id, schedules=schedules, perf=None)
                    # print("270: current_candidate.cluster_schedule", current_candidate)
                    current_candidate.perf = self._perf_func(current_candidate,
                                                             self._memory.target_params, self._value_weights)
                else:
                    current_candidate = self._memory.solution_candidate

            new_sysconf = message.working_memory.system_config
            new_candidate = message.working_memory.solution_candidate

            # Merge new information into current_sysconfig and current_candidate
            current_sysconfig = self._merge_sysconfigs(sysconfig_i=current_sysconfig, sysconfig_j=new_sysconf)
            current_candidate = self._merge_candidates(candidate_i=current_candidate,
                                                       candidate_j=new_candidate,
                                                       agent_id=self._part_id,
                                                       perf_func=self._perf_func,
                                                       target_params=self._memory.target_params,
                                                       value_weights=self._value_weights)

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
            self.print_color("start _decide", colors.BACKGROUND_LIGHT_MAGENTA)

        possible_schedules = self._schedule_provider
        # print("295: possible_schedules", possible_schedules, type(possible_schedules))
        current_best_candidate = candidate
        # print("297: candidate", candidate)
        # print("298: sysconfig", sysconfig)
        for schedule in possible_schedules:
            if self._is_local_acceptable(schedule):
                # create new candidate from sysconfig
                # print("schedule", schedule)
                new_candidate = SolutionCandidate.create_from_updated_sysconf(
                    agent_id=self._part_id, sysconfig=sysconfig, new_energy_schedule=schedule
                )
                # print("313: new_candidate", type(new_candidate), new_candidate)
                # print("313: current_best_candidate", type(current_best_candidate), current_best_candidate)
                new_candidate.perf = self._perf_func(new_candidate, self._memory.target_params, self._value_weights)
                current_best_candidate.perf = self._perf_func(current_best_candidate, self._memory.target_params, self._value_weights)
                # TODO Add the penaltys to the perf in self._perf_func() needs self._value_weights
                # only keep new candidates that perform better than the current one
                if new_candidate.perf > current_best_candidate.perf:
                    current_best_candidate = new_candidate

        schedule_in_candidate = current_best_candidate.schedules.get(self._part_id, None)
        schedule_choice_in_sysconfig = sysconfig.schedule_choices.get(self._part_id, None)

        if schedule_choice_in_sysconfig is None or \
                not np.array_equal(schedule_in_candidate, schedule_choice_in_sysconfig.energy_schedules):
            # update Sysconfig if your schedule in the current sysconf is different to the one in the candidate
            sysconfig.schedule_choices[self._part_id] = ScheduleSelection(
                energy_schedules=schedule_in_candidate, counter=self._counter + 1)
            # update counter
            self._counter += 1

        # print("326: sysconfig", sysconfig)
        # print("327: current_best_candidate", current_best_candidate)
        if test_print("start _decide"):
            self.print_color("end _decide", colors.BACKGROUND_LIGHT_MAGENTA)
        return sysconfig, current_best_candidate

    def _act(self, new_sysconfig: SystemConfig, new_candidate: SolutionCandidate) -> CohdaMessage:
        """
        Stores the new SystemConfig and SolutionCandidate in Memory and returns the COHDA message that should be sent
        :param new_sysconfig: The SystemConfig as a result from perceive and decide
        :param new_candidate: The SolutionCandidate as a result from perceive and decide
        :return: The COHDA message that should be sent
        """
        # update memory
        self._memory.system_config = new_sysconfig
        self._memory.solution_candidate = new_candidate
        # return COHDA message
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
            # An a might be in key_set_i, key_set_j or in both!
            if a in key_set_i and \
                    (a not in key_set_j or sysconfig_i_schedules[a].counter >= sysconfig_j_schedules[a].counter):
                # Use data of sysconfig_i
                schedule_selection = sysconfig_i_schedules[a]
            else:
                # Use data of sysconfig_j
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
                          agent_id: str, perf_func: Callable, target_params, value_weights):
        """
        Returns a merged Candidate. If the candidate_i remains unchanged, the same instance of candidate_i is
        returned, otherwise a new object is created with agent_id as candidate.agent_id
        :param candidate_i: The first candidate
        :param candidate_j: The second candidate
        :param agent_id: The agent_id that defines who is the creator of a new candidate
        :param perf_func: The performance function
        :param target_params: The current target parameters (e. g. a target schedule)
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
            candidate = SolutionCandidate(agent_id=agent_id, schedules=new_schedules, perf=None)
            candidate.perf = perf_func(candidate, target_params, value_weights)

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
        # cancel all cohda tasks
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
