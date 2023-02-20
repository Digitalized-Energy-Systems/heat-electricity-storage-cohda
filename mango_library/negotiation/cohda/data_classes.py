"""
Module that holds the data classes necessary for a COHDA negotiation
"""

from typing import Dict

import numpy as np
from mango.messages.codecs import json_serializable


@json_serializable
class EnergySchedules:
    """
    Model for the schedules in solution candidates in COHDA.
    """

    def __init__(self, dict_schedules: Dict[str, np.array], perf=None) -> None:
        """One dictionary for all energy schedules

        :return: None
        """
        self._dict_schedules = dict_schedules
        self._perf = perf

    def __str__(self):
        string = ""
        # string += "EnergySchedules"
        # if self.perf is not None: string += f" perf: {np.round(self.perf).astype(int)}"
        for dict_key in self.dict_schedules.keys():
            values = str(list(np.round(self.dict_schedules[dict_key], 2))).replace(', ', '\t')
            string += f"\t\'{dict_key}\': {values}"
        return string

    def __eq__(self, o: object) -> bool:
        if isinstance(o, EnergySchedules):
            if len(self.dict_schedules.keys()) == len(o.dict_schedules.keys()):
                for dict_key in self.dict_schedules.keys():
                    if dict_key in o.dict_schedules:
                        if not np.array_equal(self.dict_schedules[dict_key], o.dict_schedules[dict_key]):
                            return False
                return True
        return False

    def __add__(self, o: object):
        if isinstance(o, EnergySchedules):
            dict_schedules = self.dict_schedules.copy()
            for dict_key in o.dict_schedules.keys():
                if dict_key in dict_schedules.keys():
                    dict_schedules[dict_key] = np.add(dict_schedules[dict_key], o.dict_schedules[dict_key])
                else:
                    dict_schedules[dict_key] = o.dict_schedules[dict_key]
            perf = None
            if self.perf is not None and o.perf is not None:
                perf = self.perf - o.perf
            return EnergySchedules(dict_schedules=dict_schedules, perf=perf)
        # TODO throw Exception(__add__)
        return self

    def __sub__(self, o: object):
        if isinstance(o, EnergySchedules):
            dict_schedules = self.dict_schedules.copy()
            for dict_key in self.dict_schedules.keys():
                if dict_key in o.dict_schedules.keys():
                    dict_schedules[dict_key] = np.subtract(dict_schedules[dict_key], o.dict_schedules[dict_key])
            perf = None
            if self.perf is not None and o.perf is not None:
                perf = self.perf - o.perf
            return EnergySchedules(dict_schedules=dict_schedules, perf=perf)
        # TODO throw Exception(__sub__)
        return self

    def sum(self):
        result = {}
        for dict_key in self.dict_schedules.keys():
            result[dict_key] = np.sum(self.dict_schedules[dict_key])
        return result

    @property
    def dict_schedules(self) -> Dict[str, np.array]:
        """Return the energy schedules

        :return: dict of the energy schedules
        """
        return self._dict_schedules

    @dict_schedules.setter
    def dict_schedules(self, new_dict_schedules: Dict[str, np.array]):
        """Set the energy schedule for the agent

        :param new_dict_schedules: new_dict_schedules
        """
        self._dict_schedules = new_dict_schedules

    @property
    def perf(self) -> float:
        """Return the performance of the energy schedules

        :return: float of the energy schedules performance
        """
        return self._perf


@json_serializable
class SolutionCandidate:
    """
    Model for a solution candidate in COHDA.
    """

    def __init__(self, agent_id: str, schedules: Dict[str, EnergySchedules]) -> None:
        self._agent_id = agent_id
        self._schedules = schedules
        self._perf = None

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SolutionCandidate):
            return False
        schedules_equal = True
        if not set(self.schedules.keys()) == set(o.schedules.keys()):
            schedules_equal = False
        else:
            for k, v in self.schedules.items():
                if not np.array_equal(self.schedules[k], o.schedules[k]):
                    schedules_equal = False
        return self.agent_id == o.agent_id and self.perf == o.perf and schedules_equal

    def __str__(self):
        string = "SolutionCandidate perf: " + str(self.perf)
        for schedule_keys in self.schedules.keys():
            string += f"\n{schedule_keys} {self.schedules[schedule_keys]}"
        return string

    def to_energy_schedules(self):
        dict_schedules = {}
        for schedule_key in self.schedules.keys():
            for dict_key in self.schedules[schedule_key].dict_schedules.keys():
                if dict_key in dict_schedules.keys():
                    dict_schedules[dict_key] = np.sum([dict_schedules[dict_key], self.schedules[schedule_key].dict_schedules[dict_key]], axis=0)
                else:
                    dict_schedules[dict_key] = self.schedules[schedule_key].dict_schedules[dict_key]
        return EnergySchedules(dict_schedules=dict_schedules)

    @property
    def agent_id(self) -> str:
        """Return the agent id

        :return: agent id
        """
        return self._agent_id

    @agent_id.setter
    def agent_id(self, new_id: str):
        """Set the agent id

        :param new_id: agent id
        """
        self._agent_id = new_id

    @property
    def schedules(self) -> Dict[str, EnergySchedules]:
        """Return the candidates EnergySchedules (part_id -> Dict[str, EnergySchedules])

        :return: map part_id -> Dict[str, EnergySchedules]
        """
        return self._schedules

    @property
    def perf(self) -> float:
        """
        Returns the performance value of the candidate
        :return:
        """
        if self._perf is None:
            perf = 0
            for key in self._schedules.keys():
                perf += self._schedules[key].perf
            self._perf = perf
        return self._perf

    @perf.setter
    def perf(self, new_perf: float):
        """Set the perf

        :param new_perf: perf
        """
        self._perf = new_perf

    @property
    def cluster_schedule(self) -> EnergySchedules:
        """
        Return the candidates cluster schedule
        :return: cluster_schedule as EnergySchedules
        """
        cluster_schedule = EnergySchedules(dict_schedules={})
        for agent_energy_schedules in list(self.schedules.values()):
            for energy_schedule_key in list(agent_energy_schedules.dict_schedules.keys()):
                dict_schedules = cluster_schedule.dict_schedules
                if energy_schedule_key in dict_schedules:
                    dict_schedules[energy_schedule_key] = np.sum([dict_schedules[energy_schedule_key], agent_energy_schedules[energy_schedule_key]], axis=0)
                else:
                    dict_schedules[energy_schedule_key] = agent_energy_schedules.dict_schedules[energy_schedule_key]
                cluster_schedule.dict_schedules = dict_schedules
        return cluster_schedule

    @classmethod
    def create_from_updated_sysconf(cls, sysconfig, agent_id: str, new_energy_schedule: EnergySchedules):
        """
        Creates a Candidate based on the cluster schedule of a SystemConfiguration,
        which is changed only for *agent_id* towards *new_schedule*
        :param sysconfig: the systemconfig the candidate should be based on
        :param agent_id: the agent_id which schedule should be changed. It is also the agent_id that is the creator of
        the new Candidate
        :param new_energy_schedule: the new EnergySchedules for *agent_id*
        :return: A new SolutionCandidate object (without calculated performance!)
        """
        schedule_dict = {k: v.energy_schedules for k, v in sysconfig.schedule_choices.items()}
        schedule_dict[agent_id] = new_energy_schedule
        return cls(agent_id=agent_id, schedules=schedule_dict)


@json_serializable
class ScheduleSelection:
    """
    A selection of a specific schedule
    """

    def __init__(self, energy_schedules: EnergySchedules, counter: int) -> None:
        self._energy_schedules = energy_schedules
        self._counter = counter


    def __eq__(self, o: object) -> bool:
        return isinstance(o, ScheduleSelection) \
            and self.counter == o.counter \
            and self.energy_schedules == o.energy_schedules

    def __str__(self):
        return "ScheduleSelection: " + str(self.energy_schedules)

    @property
    def counter(self) -> int:
        """
        The counter of the selection
        :return: the counter
        """
        return self._counter

    @property
    def energy_schedules(self) -> EnergySchedules:
        """
        The schedules as EnergySchedules
        :return: energy_schedules
        """
        return self._energy_schedules


@json_serializable
class SystemConfig:
    """
    Model for a system configuration in COHDA
    """

    def __init__(self, schedule_choices: Dict[str, ScheduleSelection]) -> None:
        self._schedule_choices = schedule_choices

    def __eq__(self, o: object) -> bool:
        return isinstance(o, SystemConfig) and self._schedule_choices == o._schedule_choices

    def __str__(self):
        string = "SystemConfig"
        for schedule_keys in self.schedule_choices.keys():
            string += "\n" + schedule_keys + " " + self.schedule_choices[schedule_keys].__str__()
        return string

    @property
    def schedule_choices(self) -> Dict[str, ScheduleSelection]:
        """Return the schedule_choices map (part_id -> scheduleSelection)

        :return: Dict with part_id -> ScheduleSelection
        """
        return self._schedule_choices

    @property
    def cluster_schedule(self) -> np.array:
        """
        Return the cluster schedule of the current sysconfig
        :return: the cluster schedule as np.array
        """
        return np.array([selection.energy_schedules for selection in self.schedule_choices.values()])


@json_serializable
class WorkingMemory:
    """Working memory of a COHDA agent
    """

    def __init__(self, target_params, system_config: SystemConfig,
                 solution_candidate: SolutionCandidate):
        self._target_params = target_params
        self._system_config = system_config
        self._solution_candidate = solution_candidate

    @property
    def target_params(self):
        """Return the target parameters

        :return: the target params
        """
        return self._target_params

    @target_params.setter
    def target_params(self, new_target_params):
        """
        Set the parameters for the target
        :param new_target_params: new parameters for the target
        """
        self._target_params = new_target_params

    @property
    def system_config(self) -> SystemConfig:
        """
       The system config as SystemConfig
        :return: the believed system state
        """
        return self._system_config

    @system_config.setter
    def system_config(self, new_sysconfig: SystemConfig):
        """
        Sets the new systemconfig of the WorkingMemory
        :param new_sysconfig: the new SystemConfig object
        """
        self._system_config = new_sysconfig

    @property
    def solution_candidate(self) -> SolutionCandidate:
        """
        The current best known solution candidate for the planning
        :return: the solution candidate
        """
        return self._solution_candidate

    @solution_candidate.setter
    def solution_candidate(self, new_solution_candidate: SolutionCandidate):
        """
        Set the solution candidate
        :param new_solution_candidate: new solution candidate
        """
        self._solution_candidate = new_solution_candidate

    def __eq__(self, o: object) -> bool:
        return isinstance(o, WorkingMemory) and self.solution_candidate == o.solution_candidate \
               and self.system_config == o.system_config and self.target_params == o.target_params
