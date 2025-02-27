from mango_library.coalition.core import CoalitionModel
import pytest
import uuid
from mango.core.container import Container
from mango.role.core import RoleAgent
from mango_library.negotiation.cohda.cohda import *
from test_data import test_decide_params, test_perceive_params
import asyncio


def test_cohda_init():
    cohda = COHDA(schedule_provider=lambda: [[0, 1, 2], [1, 2, 3]],
                  is_local_acceptable=lambda s: True,
                  part_id='1')
    cohda_message = CohdaMessage(WorkingMemory(([1, 2, 3], [1, 1, 1]), SystemConfig({}), SolutionCandidate('1', {}, 0)))
    cohda._perceive([cohda_message])

    assert cohda._memory.target_params == ([1, 2, 3], [1, 1, 1])


def test_cohda_selection_multi():
    cohda = COHDA(schedule_provider=lambda: [[0, 1, 2], [1, 2, 3], [1, 1, 1], [4, 2, 3]],
                  is_local_acceptable=lambda s: True, part_id='1')
    cohda_message = CohdaMessage(WorkingMemory(([1, 2, 1], [1, 1, 1]), SystemConfig({}), SolutionCandidate('1', {}, 0)))
    sysconf, candidate = cohda._perceive([cohda_message])
    sysconf, candidate = cohda._decide(candidate=candidate, sysconfig=sysconf)

    assert np.array_equal(candidate.schedules['1'], [1, 1, 1])
    assert sysconf.schedule_choices['1'].counter == 2


@pytest.mark.parametrize('old_sysconfig, old_candidate, messages, expected_sysconfig, expected_candidate',
                         test_perceive_params
                         )
def test_perceive(old_sysconfig: SystemConfig, old_candidate: SolutionCandidate, messages: List[CohdaMessage],
                  expected_sysconfig: SystemConfig, expected_candidate: SolutionCandidate):
    cohda = COHDA(schedule_provider=lambda: [[0, 1, 2], [1, 2, 3], [1, 1, 1], [4, 2, 3]],
                  is_local_acceptable=lambda s: True, part_id='1')
    cohda._memory.system_config = old_sysconfig
    cohda._memory.solution_candidate = old_candidate
    new_sysconfig, new_candidate = cohda._perceive(messages=messages)
    for part_id in new_sysconfig.schedule_choices:
        assert np.array_equal(
            new_sysconfig.schedule_choices[part_id].energy_schedules, expected_sysconfig.schedule_choices[part_id].energy_schedules), \
            f'part_id: {part_id}, schedules: {new_sysconfig.schedule_choices[part_id].energy_schedules},' \
            f'{expected_sysconfig.schedule_choices[part_id].energy_schedules}'
        assert new_sysconfig.schedule_choices[part_id].counter == expected_sysconfig.schedule_choices[part_id].counter

        assert np.array_equal(new_candidate.schedules[part_id], expected_candidate.schedules[part_id])
    assert new_sysconfig == expected_sysconfig
    assert new_candidate.agent_id == expected_candidate.agent_id
    assert new_candidate.perf == expected_candidate.perf
    assert new_candidate == expected_candidate


@pytest.mark.parametrize('old_sysconfig, old_candidate, cohda_object, expected_sysconfig, expected_candidate',
                         test_decide_params
                         )
def test_decide(old_sysconfig: SystemConfig, old_candidate: SolutionCandidate, cohda_object: COHDA,
                expected_sysconfig: SystemConfig, expected_candidate: SolutionCandidate):

    new_sysconfig, new_candidate = cohda_object._decide(sysconfig=old_sysconfig, candidate=old_candidate)
    for part_id in new_sysconfig.schedule_choices:
        assert np.array_equal(
            new_sysconfig.schedule_choices[part_id].energy_schedules, expected_sysconfig.schedule_choices[part_id].energy_schedules), \
            f'part_id: {part_id}, schedules: {new_sysconfig.schedule_choices[part_id].energy_schedules},' \
            f'{expected_sysconfig.schedule_choices[part_id].energy_schedules}'
        assert new_sysconfig.schedule_choices[part_id].counter == expected_sysconfig.schedule_choices[part_id].counter

        assert np.array_equal(new_candidate.schedules[part_id], expected_candidate.schedules[part_id])
    assert new_sysconfig == expected_sysconfig
    assert new_candidate.agent_id == expected_candidate.agent_id
    assert new_candidate.perf == expected_candidate.perf
    assert new_candidate == expected_candidate


@pytest.mark.asyncio
async def test_optimize_simple_test_case():
    # create containers

    c = await Container.factory(addr=('127.0.0.2', 5555))

    s_array = [[[1, 1, 1, 1, 1], [4, 3, 3, 3, 3], [6, 6, 6, 6, 6], [9, 8, 8, 8, 8], [11, 11, 11, 11, 11]]]

    # create agents
    agents = []
    addrs = []
    for _ in range(10):
        a = RoleAgent(c)
        cohda_role = COHDARole(lambda: s_array[0], lambda s: True)
        a.add_role(cohda_role)
        agents.append(a)
        addrs.append((c.addr, a._aid))

    part_id = 0
    coal_id = uuid.uuid1()
    for a in agents:
        coalition_model = a._agent_context.get_or_create_model(CoalitionModel)
        coalition_model.add(coal_id, CoalitionAssignment(coal_id, list(
            filter(lambda a_t: a_t[0] != str(part_id),
                   map(lambda ad: (ad[1], c.addr, ad[0].aid),
                       zip(agents, range(10))))), 'cohda', str(part_id), 'agent_0', 1))
        part_id += 1

    agents[0].add_role(CohdaNegotiationStarterRole(([110, 110, 110, 110, 110], [1, 1, 1, 1, 1])))

    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'

    # for a in agents:
    #     await a.tasks_complete()

    # await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)
    await asyncio.sleep(0.4)

    # gracefully shutdown
    for a in agents:
        await a.shutdown()
    await c.shutdown()

    assert len(asyncio.all_tasks()) == 1
    assert np.array_equal(agents[0].roles[0]._cohda[coal_id]._memory.solution_candidate.schedules['0'],
                          np.array([11, 11, 11, 11, 11]))


@pytest.mark.asyncio
async def test_optimize_simple_test_case_multi_coal():
    # create containers

    c = await Container.factory(addr=('127.0.0.2', 5556))

    s_array = [[[1, 1, 1, 1, 1], [4, 3, 3, 3, 3], [6, 6, 6, 6, 6], [9, 8, 8, 8, 8], [11, 11, 11, 11, 11]]]

    # create agents
    agents = []
    addrs = []
    for _ in range(10):
        a = RoleAgent(c)
        cohda_role = COHDARole(lambda: s_array[0], lambda s: True)
        a.add_role(cohda_role)
        agents.append(a)
        addrs.append((c.addr, a._aid))

    part_id = 0
    coal_id = uuid.uuid1()
    coal_id2 = uuid.uuid1()
    for a in agents:
        coalition_model = a._agent_context.get_or_create_model(CoalitionModel)
        coalition_model.add(coal_id, CoalitionAssignment(coal_id, [], 'cohda', str(part_id), 'agent_0', 1))
        coalition_model.add(coal_id2, CoalitionAssignment(coal_id2, list(
            filter(lambda a_t: a_t[0] != part_id,
                   map(lambda ad: (ad[1], c.addr, ad[0].aid),
                       zip(agents, range(10))))), 'cohda', str(part_id), 'agent_0', 1))
        part_id += 1

    # as the coalition assignment 0 does not contain the correct participants
    # the correct assignment is exactly chosen when
    # the solution candidate is correct due to the agents schedules and the target
    agents[0].add_role(
        CohdaNegotiationStarterRole(([110, 110, 110, 110, 110], [1, 1, 1, 1, 1]), coalition_uuid=coal_id2))

    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'

    await asyncio.sleep(0.4)
    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)

    # gracefully shutdown
    for a in agents:
        await a.shutdown()
    await c.shutdown()

    assert len(asyncio.all_tasks()) == 1
    assert np.array_equal(agents[0].roles[0]._cohda[coal_id2]._memory.solution_candidate.schedules['0'],
                          [11, 11, 11, 11, 11])


@pytest.mark.asyncio
async def test_optimize_hinrichs_test_case():
    # create containers

    c = await Container.factory(addr=('127.0.0.2', 5557))

    s_array = [[[1, 1, 1, 1, 1], [4, 3, 3, 3, 3], [6, 6, 6, 6, 6], [9, 8, 8, 8, 8], [11, 11, 11, 11, 11]],
               [[13, 12, 12, 12, 12], [15, 15, 15, 14, 14], [18, 17, 17, 17, 17], [20, 20, 20, 19, 19],
                [23, 22, 22, 22, 22]],
               [[25, 24, 23, 23, 23], [27, 26, 26, 25, 25], [30, 29, 28, 28, 28], [32, 31, 31, 30, 30],
                [35, 34, 33, 33, 33]],
               [[36, 35, 35, 34, 34], [39, 38, 37, 36, 36], [41, 40, 40, 39, 39], [44, 43, 42, 41, 41],
                [46, 45, 45, 44, 44]],
               [[48, 47, 46, 45, 45], [50, 49, 48, 48, 47], [53, 52, 51, 50, 50], [55, 54, 53, 53, 52],
                [58, 57, 56, 55, 55]],
               [[60, 58, 57, 56, 56], [62, 61, 60, 59, 58], [65, 63, 62, 61, 61], [67, 66, 65, 64, 63],
                [70, 68, 67, 66, 66]],
               [[71, 70, 68, 67, 67], [74, 72, 71, 70, 69], [76, 75, 73, 72, 72], [79, 77, 76, 75, 74],
                [81, 80, 78, 77, 77]],
               [[83, 81, 80, 78, 78], [85, 83, 82, 81, 80], [88, 86, 85, 83, 83], [90, 88, 87, 86, 85],
                [93, 91, 90, 88, 88]],
               [[95, 92, 91, 90, 89], [97, 95, 93, 92, 91], [100, 97, 96, 95, 94], [102, 100, 98, 97, 96],
                [105, 102, 101, 100, 99]],
               [[106, 104, 102, 101, 100], [109, 106, 105, 103, 102], [111, 109, 107, 106, 105],
                [114, 111, 110, 108, 107], [116, 114, 112, 111, 110]]]

    # create agents
    agents = []
    addrs = []
    for i in range(10):
        a = RoleAgent(c)
        cohda_role = COHDARole(lambda n=i: s_array[n], lambda s: True)
        a.add_role(cohda_role)
        agents.append(a)
        addrs.append((c.addr, a._aid))

    part_id = 0
    coal_id = uuid.uuid1()
    for a in agents:
        coalition_model = a._agent_context.get_or_create_model(CoalitionModel)
        coalition_model.add(coal_id, CoalitionAssignment(coal_id, list(
            filter(lambda a_t: a_t[0] != str(part_id),
                   map(lambda ad: (ad[1], c.addr, ad[0].aid),
                       zip(agents, range(10))))), 'cohda', str(part_id), 'agent_0', 1))
        part_id += 1

    agents[0].add_role(CohdaNegotiationStarterRole(([542, 528, 519, 511, 509], [1, 1, 1, 1, 1])))

    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                raise a._check_inbox_task.exception()
            else:
                assert False, f'check_inbox terminated unexpectedly.'

    await asyncio.sleep(0.4)

    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)

    # gracefully shutdown
    for a in agents:
        await a.shutdown()
    await c.shutdown()

    assert len(asyncio.all_tasks()) == 1
    cluster_schedule = np.array(list(
        map(lambda item: item[1], agents[0].roles[0]._cohda[coal_id]._memory.solution_candidate.schedules.items())))
    assert [543, 529, 520, 512, 510] == cluster_schedule.sum(axis=0).tolist()


async def wait_for_coalition_built(agents):
    for agent in agents:
        while not agent.inbox.empty():
            await asyncio.sleep(0.1)
