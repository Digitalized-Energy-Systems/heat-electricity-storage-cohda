import uuid

import pandas as pd
from mango.core.container import Container
from mango.role.core import RoleAgent

from mango_library.coalition.core import CoalitionModel
from mango_library.negotiation.cohda.cohda import *
from mango_library.negotiation.termination import NegotiationTerminationRole

addr = ('127.0.0.2', 5557)

" Pandas "
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)
pd.options.display.float_format = '{:,.0f}'.format
pd.set_option('display.colheader_justify', 'center')


async def test_case(power_target, heat_target, value_weights, schedules_provider):
    c = await Container.factory(addr=addr)

    global_start_time = time.time()
    t = time.localtime(time.time())
    filename = f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}_{t.tm_hour:02d}-{t.tm_min:02d}-{t.tm_sec:02d}"
    for value_weight in value_weights:
        value_weight["global_start_time"] = global_start_time
        value_weight["filename"] = filename
    f = open(filename, "a")
    f.write(f"{power_target}\n")
    f.write(f"{heat_target}\n")
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
    " calc best_energy_schedules "
    best_solution_candidate = None
    best_solution_candidate_perf = float("-inf")
    print(f"{Colors.BOLD}Search for best_solution_candidate{Colors.RESET_ALL}")
    for agent in agents:
        solution_candidate = agent.roles[0]._cohda[coal_id]._best_solution_candidate
        print(f"{agent.aid} - perf: {np.round(solution_candidate.perf, 2)}")
        if best_solution_candidate is None or solution_candidate.perf < best_solution_candidate_perf:
            best_solution_candidate = solution_candidate
            best_solution_candidate_perf = solution_candidate.perf
    print(f"BEST: {best_solution_candidate}")

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
    print(pd.DataFrame(data=data, columns=pd.MultiIndex.from_frame(pd.DataFrame(data=columns), names=["", ""])))
    coalition_sum = np.sum(np.abs(data), axis=0)
    print(pd.DataFrame(data=[coalition_sum], columns=pd.MultiIndex.from_frame(pd.DataFrame(data=columns), names=["", ""])))
    print(f"{coalition_sum[1]:.2f}*{value_weights[0]['power_kwh_price']}", end=" + ")
    result = coalition_sum[1] * value_weights[0]['power_kwh_price']
    print(f"{coalition_sum[4]:.2f}*{value_weights[0]['heat_kwh_price']}", end=" + ")
    result += coalition_sum[4] * value_weights[0]['heat_kwh_price']
    print(f"{coalition_sum[6]:.2f}*{value_weights[0]['gas_price']}", end=" - ")
    result += coalition_sum[6] * value_weights[0]['gas_price']
    print(f"{coalition_sum[8]:.2f}*{value_weights[0]['converted_price']}", end=" = ")
    result += coalition_sum[8] * value_weights[0]['converted_price']
    print(f"{result:.2f}")
    print(f"DIFF: {int(coalition_sum[2])}+{int(coalition_sum[5])} = {int(coalition_sum[2] + coalition_sum[5])}")

    global print_data
    i = 0
    for part in print_data:
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


def get_wind_schedule(maxv, number, sigma, cut, power):
    day = []
    value = 0
    for i in range(96):
        r = np.random.normal(0, pow(sigma, 0.5))
        for j in range(2):
            value = abs(min(maxv, value + r))
            day.append(int(pow(value, 3) / 10))
    schedule = get_schedule(np.array(day[96:]), number, power, cut)
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


# avg = []
# maxv = []
# for i in range(10000):
#     _, avg_, maxv_ = get_wind_schedule(36)
#     avg.append(avg_)
#     maxv.append(maxv_)
# print(f"{int(1000*sum(avg) / len(avg))/1000}\t{avg}")
# print(f"{int(sum(maxa) / len(maxa))}\t{min(maxa)}\t{max(maxa)}\t{maxa}")
# print(f"{sum(avg) / len(avg)}")
# print(f"{int(sum(maxv) / len(maxv))}\t{min(maxv)}\t{max(maxv)}")
# print(f"{sum(day)}\t{min(day)}\t{max(day)}\t{int(sum(day)/len(day))}\t{day}")
# res = []
# for i in day:
#     res.append(i * i * i)
# print(min(res), max(res), int(sum(res)/len(res)), res)

power_penalty = .069261538461538464661604308503228821825
power_penalty = 10000000
# heat_penalty = power_penalty
heat_penalty = 10
power_kwh_price = .15
heat_kwh_price = .1
converted_price = .05
maximum_agent_attempts = 5
# 'convert_amount', 'gas_price', 'max_gas_amount', 'gas_to_heat_factor', 'gas_to_power_factor', 'power_to_heat_factor', 'power_to_heat_amount'
val = [
    [0, 999 * 0.11 / 430, 430, 951 / 430, 999 / 430, .9, 1000, [np.zeros(96)]],
    [0, 999 * 0.11 / 430, 430, 951 / 430, 999 / 430, .9, 1000, [np.zeros(96)]],
    [0, 999 * 0.11 / 430, 430, 951 / 430, 999 / 430, .9, 1000, [np.zeros(96)]],
    [0, 999 * 0.11 / 430, 430, 951 / 430, 999 / 430, .9, 1000, [np.zeros(96)]],
    [0, 999 * 0.11 / 430, 430, 951 / 430, 999 / 430, .9, 1000, [np.zeros(96)]],
    [0, 999 * 0.11 / 430, 430, 951 / 430, 999 / 430, .9, 1000, [np.zeros(96)]],
    [0, 999 * 0.11 / 430, 430, 951 / 430, 999 / 430, .9, 1000, [np.zeros(96)]],
    [0, 999 * 0.11 / 430, 430, 951 / 430, 999 / 430, .9, 1000, [np.zeros(96)]],
    [0, 999 * 0.11 / 430, 430, 951 / 430, 999 / 430, .9, 1000, [np.zeros(96)]],
    # [0, 0, 0, 0, 0, .9, 10000, [get_wind_schedule(36, 5, 3.7, .1, 1)[0]]],
    # [0, 0, 0, 0, 0, .9, 10000, [get_wind_schedule(36, 5, 3.7, .1, 2)[0]]],
    # [0, 0, 0, 0, 0, .9, 10000, [get_wind_schedule(36, 5, 3.7, .1, 3)[0]]],
    # [0, 0, 0, 0, 0, .9, 10000, [get_wind_schedule(36, 5, 3.7, .1, 2)[0]]],
    # [0, 0, 0, 0, 0, .9, 10000, [get_wind_schedule(36, 5, 3.7, .1, 1)[0]]],
]
power_target = [500, 520, 520, 520, 520, 520, 520, 520, 620, 720, 820, 3200, 4200, 5200, 8500, 12520, 8520, 4420, 2520, 1520, 520, 520, 520, 520, 80, 130, 220, 280, 470, 400, 440, 550, 570, 820, 900, 1010, 1230, 1520, 2400, 3000, 3400, 3800, 4200, 4600, 5000, 5400, 4800, 6000, 6100, 6200, 6300, 6400, 6400, 6400, 6500, 6600, 6600, 6500, 6500, 6500, 6400, 6100, 6100, 6200, 6100, 5800, 5600, 5100, 4200, 4600, 4700, 3300, 3300, 1860, 1270, 820, 540, 376, 220, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520],
heat_target = [3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 2950, 2900, 2850, 2800, 2750, 2700, 2650, 2600, 2550, 2500, 2450, 2400, 2350, 2300, 2250, 2200, 2150, 2100, 2050, 2000, 1950, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000],
power_target = power_target[0]
heat_target = heat_target[0]
# power_target = np.ones(96)
# heat_target = np.ones(96)
power_target = power_target / np.sum(power_target)
print(power_target)
power_target = (power_target * 9.6E4 * 4).astype(int)
heat_target = heat_target / np.sum(heat_target)
print(heat_target)
heat_target = (heat_target * 9.6E4 * 3).astype(int)

value_weights = []
schedules_provider = []
for v in val:
    value_weights.append({'convert_amount': v[0], 'gas_price': v[1], 'max_gas_amount': v[2], 'gas_to_heat_factor': v[3], 'gas_to_power_factor': v[4], 'power_to_heat_factor': v[5], 'power_to_heat_amount': v[6], 'power_penalty': power_penalty, 'heat_penalty': heat_penalty, 'power_kwh_price': power_kwh_price, 'heat_kwh_price': heat_kwh_price, 'converted_price': converted_price, 'penalty_exponent': penalty_exponent, 'max_iterations': max_iterations, 'maximum_agent_attempts': maximum_agent_attempts})
    schedules_provider.append(v[7])

asyncio.run(test_case(
    power_target=power_target,
    heat_target=heat_target,
    value_weights=value_weights,
    schedules_provider=schedules_provider,
))
