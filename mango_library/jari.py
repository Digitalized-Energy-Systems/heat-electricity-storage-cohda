import os.path
import uuid

import pandas as pd
from mango.core.container import Container
from mango.role.core import RoleAgent

from mango_library.coalition.core import CoalitionModel
from mango_library.negotiation.cohda.cohda import *
from mango_library.negotiation.termination import NegotiationTerminationRole

addr = ('127.0.0.2', random.randint(5557, 9999))

" Pandas "
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.0f}'.format
pd.set_option('display.colheader_justify', 'center')


async def test_case(power_target, heat_target, value_weights, schedules_provider):
    c = await Container.factory(addr=addr)

    global_start_time = time.time()
    t = time.localtime(time.time())
    if not os.path.exists("log"):
        os.makedirs("log")
    filename = f"log/{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}_{t.tm_hour:02d}-{t.tm_min:02d}-{t.tm_sec:02d}"
    for value_weight in value_weights:
        value_weight["global_start_time"] = global_start_time
        value_weight["filename"] = filename
    f = open(f'{filename}_start.txt', "a")
    f.write(f"{power_target}\n")
    f.write(f"{heat_target}\n")
    np.set_printoptions(linewidth=np.inf)
    for i in range(len(value_weights)):
        f.write(f"{i}: {value_weights[i]} schedules: {str(schedules_provider[i])}\n")
    f.close()

    " create agents "
    agents = []
    addrs = []
    for i, _ in enumerate(schedules_provider):
        a = RoleAgent(c)
        cohda_role = COHDARole(schedules_provider[i], value_weights[i], lambda s: True)
        a.add_role(cohda_role)
        if i == 0:
            a.add_role(CohdaNegotiationStarterRole(({'power': np.array(power_target), 'heat': np.array(heat_target)}, np.ones(len(schedules_provider[0][0])))))
        a.add_role(NegotiationTerminationRole(i == 0))
        agents.append(a)
        addrs.append((c.addr, a._aid))
    coal_id = uuid.uuid1()
    for part_id, a in enumerate(agents):
        coalition_model = a._agent_context.get_or_create_model(CoalitionModel)
        coalition_model.add(coal_id, CoalitionAssignment(coal_id, list(
            filter(lambda a_t: a_t[0] != str(part_id), map(lambda ad: (ad[1], c.addr, ad[0].aid), zip(agents, range(1000))))), 'cohda', str(part_id), 'agent0', addr))
    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                print("raise a._check_inbox_task.exception()")
            else:
                print(False, f'check_inbox terminated unexpectedly.')

    await asyncio.sleep(.4)
    await asyncio.wait_for(wait_for_term(agents), timeout=1E10)

    " gracefully shutdown "
    for a in agents:
        await a.shutdown()
    await c.shutdown()

    print()
    print()
    f = open(f'{filename}_result.txt', "a")
    " calc best_energy_schedules "
    best_solution_candidate = None
    best_solution_candidate_perf = float("-inf")
    print(f"{Colors.BOLD}Search for best_solution_candidate{Colors.RESET_ALL}")
    f.write(f"Search for best_solution_candidate\n")
    for agent in agents:
        solution_candidate = agent.roles[0]._cohda[coal_id]._best_solution_candidate
        print(f"{agent.aid} - perf: {np.round(solution_candidate.perf, 2)}")
        f.write(f"{agent.aid} - perf: {np.round(solution_candidate.perf, 2)}\n")
        if best_solution_candidate is None or solution_candidate.perf < best_solution_candidate_perf:
            best_solution_candidate = solution_candidate
            best_solution_candidate_perf = solution_candidate.perf
    print(f"BEST: {best_solution_candidate}")
    f.write(f"BEST: {best_solution_candidate}\n")

    soll = EnergySchedules(dict_schedules={'power': power_target, 'heat': heat_target})
    ist = best_solution_candidate.to_energy_schedules()
    diff = ist - soll
    columns = [('', 'SOLL'), ('Power', 'IST'), ('', 'DIFF'),
               ('', 'SOLL'), ('Heat', 'IST'), ('', 'DIFF'),
               ('', 'gas_amount'), ('IST', 'power_to_heat'), ('', 'power_to_conversion'),
               ]
    data = []
    for i, _ in enumerate(soll.dict_schedules[list(soll.dict_schedules.keys())[0]]):
        new_line = []
        " Power "
        new_line.append(soll.dict_schedules["power"][i])
        new_line.append(ist.dict_schedules["power"][i])
        new_line.append(diff.dict_schedules["power"][i])
        " Heat "
        new_line.append(soll.dict_schedules["heat"][i])
        new_line.append(ist.dict_schedules["heat"][i])
        new_line.append(diff.dict_schedules["heat"][i])
        " IST "
        new_line.append(ist.dict_schedules["gas_amount"][i])
        new_line.append(ist.dict_schedules["power_to_heat"][i])
        new_line.append(ist.dict_schedules["power_to_conversion"][i])
        data.append(new_line)

    datatable1 = pd.DataFrame(data=data, columns=pd.MultiIndex.from_frame(pd.DataFrame(data=columns), names=["", ""]))
    print(datatable1)
    f.write(f"{datatable1}\n")
    coalition_sum = np.sum(np.abs(data), axis=0)
    datatable2 = pd.DataFrame(data=[coalition_sum], columns=pd.MultiIndex.from_frame(pd.DataFrame(data=columns), names=["", ""]))
    print(datatable2)
    f.write(f"{datatable2}\n")
    calculation = f"{coalition_sum[1]:.2f}*{value_weights[0]['power_kwh_price']} + "
    calculation += f"{coalition_sum[4]:.2f}*{value_weights[0]['heat_kwh_price']} + "
    calculation += f"{coalition_sum[6]:.2f}*{value_weights[0]['gas_price']} + "
    calculation += f"{coalition_sum[8]:.2f}*{value_weights[0]['converted_price']}"
    result = coalition_sum[1] * value_weights[0]['power_kwh_price']
    result += coalition_sum[4] * value_weights[0]['heat_kwh_price']
    result += coalition_sum[6] * value_weights[0]['gas_price']
    result += coalition_sum[8] * value_weights[0]['converted_price']
    print(f"{calculation} = {result:.2f}")
    f.write(f"{calculation} = {result:.2f}\n")
    print(f"DIFF: {int(coalition_sum[2])}+{int(coalition_sum[5])} = {int(coalition_sum[2] + coalition_sum[5])}")
    f.write(f"DIFF: {int(coalition_sum[2])}+{int(coalition_sum[5])} = {int(coalition_sum[2] + coalition_sum[5])}\n")
    f.close()

    global print_data
    i = 0
    # for part in print_data:
    part = 'all'
    f = open(f'{filename}_{part}.txt', "a")
    f.write(print_data[part])
    f.close()
    i += 1

    return {'ziel': soll, 'ist': ist, 'time': time.time() - global_start_time}


async def wait_for_term(agents):
    def get_negotiation_termination_controller_role():
        for agent in agents:
            for role in agent.roles:
                if type(role) is NegotiationTerminationRole:
                    if role._is_controller:
                        return role

    negotiation_termination_controller_role = get_negotiation_termination_controller_role()
    await asyncio.sleep(.1)
    i = 0
    while i < len(agents):
        # if agents[i].inbox.empty() or float(next(iter(negotiation_termination_controller_role._weight_map.values()))) == 1:
        if agents[i].inbox.empty():
            i += 1
        else:
            i = 0
            await asyncio.sleep(0.1)


""" test wind """


def get_wind_schedule(maxv, count, sigma, cut, power):
    day = []
    value = 0
    for i in range(96):
        r = np.random.normal(0, pow(sigma, 0.5))
        for j in range(2):
            value = abs(min(maxv, value + r))
            day.append(int(pow(value, 3) / 10))
    schedule = get_schedule(np.array(day[96:]), count, power, cut)
    return schedule


def get_solar_schedule(count, cut, power):
    solar = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0001874580058407, 0.0015270868975554, 0.0026769964211354, 0.0038257745591754, 0.0061418349571946, 0.0155419124925599, 0.0237460278568065, 0.031964643380709, 0.0429912213760122, 0.0715450950470349, 0.0971256513634032, 0.122697148096535, 0.150560112092094, 0.1925885579647, 0.232222917218015, 0.271852182773238, 0.314806732042316, 0.375511516312298, 0.433441435829143, 0.491382331775775, 0.547338878050542, 0.596264492107872, 0.645968936469177, 0.695665180059844, 0.741288752163066, 0.775538313063295, 0.810597852509958, 0.845658174797026, 0.87660546038796, 0.893178679531221, 0.911289500183215, 0.929394495297822, 0.9375955255487, 0.887934082343557, 0.847787743267431, 0.807636812553217, 0.767222329360787, 0.710587195620888, 0.657877207477716, 0.605176045125917, 0.552662913274611, 0.494321271575053, 0.437541931325793, 0.380761578344725, 0.326899847542852, 0.28036173039361, 0.233445650911756,
             0.186543663694167, 0.144005325120327, 0.115422362911761, 0.0855336631206063, 0.0556500724410242, 0.0300357429549431, 0.0238133529854286, 0.0148885833587954, 0.0059516342467003, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    schedule = get_schedule(np.array(solar), count, power, cut)
    return schedule


def get_schedule(day, count, power, cut):
    schedule = []
    for _ in range(count):
        schedule.append(list(day * power * (1 - np.random.random(96) * cut)))
    return schedule


max_iterations = 2
max_iteration_power = 0
penalty_exponent = 2
power_penalty = 1
heat_penalty = 1
power_kwh_price = .15
heat_kwh_price = .1
converted_price = .05
maximum_agent_attempts = 5
# 'convert_amount', 'gas_price', 'max_gas_amount', 'gas_to_heat_factor', 'gas_to_power_factor', 'power_to_heat_factor', 'power_to_heat_amount'
val = [
    [150, 999 * 0.11 / 430, 125, 951 / 430, 999 / 430, 0, 0, 0],
    [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    [150, 999 * 0.11 / 430, 125, 951 / 430, 999 / 430, 0, 0, 0],
    [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    [150, 999 * 0.11 / 430, 125, 951 / 430, 999 / 430, 0, 0, 0],
    [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    [150, 999 * 0.11 / 430, 125, 951 / 430, 999 / 430, 0, 0, 0],
    [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    # [150, 999 * 0.11 / 430, 125, 951 / 430, 999 / 430, 0, 0, 0],
    # [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    # [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    # [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    # [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    # [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    # [150, 999 * 0.11 / 430, 125, 951 / 430, 999 / 430, 0, 0, 0],
    # [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    # [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    # [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    # [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    # [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    # [150, 999 * 0.11 / 430, 125, 951 / 430, 999 / 430, 0, 0, 0],
    # [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    # [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    # [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    # [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    # [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, .3, 200)],
    # [150, 999 * 0.11 / 430, 125, 951 / 430, 999 / 430, 0, 0, 0],
    # [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    # [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    # [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    # [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
    # [12, 0, 0, 0, 0, .9, 75, get_wind_schedule(36, 5, 3.7, 0.1, .03)],
]
power_target = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 550, 600, 650, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1700, 1700, 1700, 1650, 1600, 1550, 1500, 1450, 1400, 1350, 1300, 1300, 1300, 1500, 1700, 1900, 2100, 2300, 2500, 2500, 2500, 2450, 2400, 2350, 2300, 2250, 2200, 2150, 2100, 2050, 2000, 1950, 1900, 1850, 1850, 1850, 1850, 1900, 2000, 2100, 2200, 2200, 2200, 2100, 2000, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1800, 1700, 1500, 1300, 1100, 900, 700, 500],
heat_target = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 2950, 2900, 2850, 2800, 2750, 2700, 2650, 2600, 2550, 2500, 2450, 2400, 2350, 2300, 2250, 2200, 2150, 2100, 2050, 2000, 1950, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000],
power_target = power_target[0] + np.ones(96) * 500
heat_target = heat_target[0]
# power_target = np.ones(96)
# heat_target = np.ones(96)
power_target = power_target / np.sum(power_target)
print(f"power_target: {power_target}")
power_target = (power_target * 96E3 * 1).astype(int)
heat_target = heat_target / np.sum(heat_target)
print(f"heat_target: {heat_target}")
heat_target = (heat_target * 96E3 * 1).astype(int)

value_weights = []
schedules_provider = []
for v in val:
    value_weights.append({'convert_amount': v[0], 'gas_price': v[1], 'max_gas_amount': v[2], 'gas_to_heat_factor': v[3], 'gas_to_power_factor': v[4], 'power_to_heat_factor': v[5], 'power_to_heat_amount': v[6], 'power_penalty': power_penalty, 'heat_penalty': heat_penalty, 'power_kwh_price': power_kwh_price, 'heat_kwh_price': heat_kwh_price, 'converted_price': converted_price, 'penalty_exponent': penalty_exponent, 'max_iterations': max_iterations, 'maximum_agent_attempts': maximum_agent_attempts, 'max_iteration_power': max_iteration_power})
    if v[7] == 0:
        schedules_provider.append([np.zeros(96).tolist()])
    else:
        schedules_provider.append(v[7])

asyncio.run(test_case(
    power_target=power_target,
    heat_target=heat_target,
    value_weights=value_weights,
    schedules_provider=schedules_provider,
))
