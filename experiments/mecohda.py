import os.path
import uuid, random, asyncio, time

import pandas as pd
from mango.core.container import Container
from mango.role.core import RoleAgent

from mango_library.coalition.core import *
from mango_library.negotiation.cohda.cohda import *
import mango_library.negotiation.cohda.cohda as cohda

from mango_library.negotiation.termination import NegotiationTerminationRole
from mango_library.negotiation.cohda.data_classes import EnergySchedules

import numpy as np

DEFAULT_PENALTY_EXP = 2
LOW_PENALTY_EXP = 1
DEFAULT_PENALTY = 2

addr = ("127.0.0.2", random.randint(5557, 9999))

" Pandas "
pd.set_option("display.width", None)
pd.set_option("display.max_rows", 100)
pd.options.display.float_format = "{:,.0f}".format
pd.set_option("display.colheader_justify", "center")

POWER_TARGET = (
    [
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        550,
        600,
        650,
        700,
        800,
        900,
        1000,
        1100,
        1200,
        1300,
        1400,
        1500,
        1600,
        1700,
        1700,
        1700,
        1700,
        1650,
        1600,
        1550,
        1500,
        1450,
        1400,
        1350,
        1300,
        1300,
        1300,
        1500,
        1700,
        1900,
        2100,
        2300,
        2500,
        2500,
        2500,
        2450,
        2400,
        2350,
        2300,
        2250,
        2200,
        2150,
        2100,
        2050,
        2000,
        1950,
        1900,
        1850,
        1850,
        1850,
        1850,
        1900,
        2000,
        2100,
        2200,
        2200,
        2200,
        2100,
        2000,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1800,
        1700,
        1500,
        1300,
        1100,
        900,
        700,
        500,
    ],
)
HEAT_TARGET = (
    [
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        3000,
        2950,
        2900,
        2850,
        2800,
        2750,
        2700,
        2650,
        2600,
        2550,
        2500,
        2450,
        2400,
        2350,
        2300,
        2250,
        2200,
        2150,
        2100,
        2050,
        2000,
        1950,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1950,
        2000,
        2050,
        2100,
        2150,
        2200,
        2250,
        2300,
        2350,
        2400,
        2450,
        2500,
        2550,
        2600,
        2650,
        2700,
        2750,
        2800,
        2850,
        2900,
        2950,
        3000,
    ],
)

def sample_schedule(provider, value_weight_dict):
    if value_weight_dict["name"] == "SOLAR":
        return EnergySchedules(
                dict_schedules={
                    "power": provider[0],
                    "heat": np.zeros(96).tolist(),
                }
            )
    elif value_weight_dict["name"] == "CHP":
        schedule_power = []
        schedule_heat = []
        for _ in range(96):
            calcs = calculate_amount(value_weight_dict, 
                                             random.random()*value_weight_dict["max_gas_amount"], 
                                             dict(max_amount=1,opportunity=1,power_schedule_timestamp=1,heat_open=0,power_open=0))
            schedule_power.append(calcs["end_power"])
            schedule_heat.append(calcs["end_heat"])
        return EnergySchedules(dict_schedules={
            "power" : schedule_power,
            "heat": schedule_heat
        })

def sample_schedules(provider, value_weight_dict):
    schedules = []
    for i in range(1000):
        schedules.append(sample_schedule(provider, value_weight_dict))
    return schedules

async def test_case(
    power_target, heat_target, value_weights, schedules_provider, storages, name, use_classic=False
):
    c = await Container.factory(addr=addr)

    global_start_time = time.time()
    t = time.localtime(time.time())
    if not os.path.exists("log"):
        os.makedirs("log")
    filename = f"log/{name}-{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}_{t.tm_hour:02d}-{t.tm_min:02d}-{t.tm_sec:02d}"
    for value_weight in value_weights:
        value_weight["global_start_time"] = global_start_time
        value_weight["filename"] = filename
    f = open(f"{filename}_start.txt", "a")
    f.write(f"{power_target}\n")
    f.write(f"{heat_target}\n")
    np.set_printoptions(linewidth=np.inf)
    for i in range(len(value_weights)):
        f.write(f"{i}: {value_weights[i]} schedules: {str(schedules_provider[i])}\n")
    f.close()

    start_time = time.time()

    " create agents "
    agents = []
    addrs = []
    for i, _ in enumerate(schedules_provider):
        a = RoleAgent(c)
        cohda_role = COHDARole(schedules_provider[i], value_weights[i], 
                               is_classic=use_classic, 
                               energy_schedules=sample_schedules(schedules_provider[i], value_weights[i]) if use_classic else None)
        a.add_role(cohda_role)
        if i == 0:
            a.add_role(
                CohdaNegotiationStarterRole(
                    (
                        {
                            "power": np.array(power_target),
                            "heat": np.array(heat_target),
                        },
                        np.ones(len(schedules_provider[0][0])),
                    )
                )
            )
        a.add_role(NegotiationTerminationRole(i == 0))
        agents.append(a)
        addrs.append((c.addr, a._aid))

    # storages
    for i, storage in enumerate(storages):
        a = RoleAgent(c)
        cohda_role = COHDARole(
            schedules_provider[i],
            value_weights=value_weights[0],
            storage=storage,
            is_storage=True,
        )
        a.add_role(cohda_role)
        a.add_role(NegotiationTerminationRole(i == 0))
        agents.append(a)
        addrs.append((c.addr, a._aid))

    coal_id = uuid.uuid1()
    for part_id, a in enumerate(agents):
        coalition_model = a._agent_context.get_or_create_model(CoalitionModel)
        coalition_model.add(
            coal_id,
            CoalitionAssignment(
                coal_id,
                list(
                    filter(
                        lambda a_t: a_t[0] != str(part_id),
                        map(
                            lambda ad: (ad[1], c.addr, ad[0].aid),
                            zip(agents, range(1000)),
                        ),
                    )
                ),
                "cohda",
                str(part_id),
                "agent0",
                addr,
            ),
        )
    for a in agents:
        if a._check_inbox_task.done():
            if a._check_inbox_task.exception() is not None:
                print("raise a._check_inbox_task.exception()")
            else:
                print(False, f"check_inbox terminated unexpectedly.")

    await asyncio.sleep(0.4)
    await asyncio.wait_for(wait_for_term(agents), timeout=1e10)

    " gracefully shutdown "
    for a in agents:
        await a.shutdown()
    await c.shutdown()

    time_elapsed = time.time() - start_time

    pd.DataFrame([{"scenario": name, "time": time_elapsed}]).to_csv(
        f"{filename}_time_df.csv"
    )

    print()
    print()
    f = open(f"{filename}_result.txt", "a")
    " calc best_energy_schedules "
    best_solution_candidate = None
    best_solution_candidate_perf = float("-inf")
    f.write(f"Search for best_solution_candidate\n")
    for agent in agents:
        solution_candidate = agent.roles[0]._cohda[coal_id]._best_solution_candidate
        print(f"{agent.aid} - perf: {np.round(solution_candidate.perf, 2)}")
        f.write(f"{agent.aid} - perf: {np.round(solution_candidate.perf, 2)}\n")
        if (
            best_solution_candidate is None
            or solution_candidate.perf < best_solution_candidate_perf
        ):
            best_solution_candidate = solution_candidate
            best_solution_candidate_perf = solution_candidate.perf
    print(f"BEST: {best_solution_candidate}")
    f.write(f"BEST: {best_solution_candidate}\n")

    bsc_expanded: pd.DataFrame = best_solution_candidate.dataframe_expanded()
    bsc_expanded["agent_type"] = (
        bsc_expanded["agent"]
        .astype(int)
        .apply(
            lambda v: (
                "STORAGE" if v >= len(value_weights) else value_weights[v]["name"]
            ),
        )
    )

    bsc_expanded.to_csv(f"{filename}_result_df.csv")

    soll = EnergySchedules(dict_schedules={"power": power_target, "heat": heat_target})
    ist = best_solution_candidate.to_energy_schedules()
    diff = ist - soll
    columns = [
        ("", "SOLL"),
        ("Power", "IST"),
        ("", "DIFF"),
        ("", "SOLL"),
        ("Heat", "IST"),
        ("", "DIFF"),
        ("", "gas_amount"),
        ("IST", "power_to_heat"),
        ("", "power_to_conversion"),
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

    datatable1 = pd.DataFrame(
        data=data,
        columns=pd.MultiIndex.from_frame(pd.DataFrame(data=columns), names=["", ""]),
    )
    print(datatable1)
    f.write(f"{datatable1}\n")

    datatable1.to_csv(f"{filename}_result_df_cs.csv")
    coalition_sum = np.sum(np.abs(data), axis=0)
    datatable2 = pd.DataFrame(
        data=[coalition_sum],
        columns=pd.MultiIndex.from_frame(pd.DataFrame(data=columns), names=["", ""]),
    )
    print(datatable2)
    f.write(f"{datatable2}\n")
    calculation = f"{coalition_sum[1]:.2f}*{value_weights[0]['power_kwh_price']} + "
    calculation += f"{coalition_sum[4]:.2f}*{value_weights[0]['heat_kwh_price']} + "
    calculation += f"{coalition_sum[6]:.2f}*{value_weights[0]['gas_price']} + "
    calculation += f"{coalition_sum[8]:.2f}*{value_weights[0]['converted_price']}"
    result = coalition_sum[1] * value_weights[0]["power_kwh_price"]
    result += coalition_sum[4] * value_weights[0]["heat_kwh_price"]
    result += coalition_sum[6] * value_weights[0]["gas_price"]
    result += coalition_sum[8] * value_weights[0]["converted_price"]
    print(f"{calculation} = {result:.2f}")
    f.write(f"{calculation} = {result:.2f}\n")
    print(
        f"DIFF: {int(coalition_sum[2])}+{int(coalition_sum[5])} = {int(coalition_sum[2] + coalition_sum[5])}"
    )
    f.write(
        f"DIFF: {int(coalition_sum[2])}+{int(coalition_sum[5])} = {int(coalition_sum[2] + coalition_sum[5])}\n"
    )
    f.close()

    i = 0
    # for part in print_data:
    part = "all"
    f = open(f"{filename}_{part}.txt", "a")
    f.write(cohda.print_data[part])
    f.close()
    i += 1

    print_data_all_rows = cohda.print_data[part].split("\n")[:-1]
    df_rows = []
    for row in print_data_all_rows:
        cells = row.split("\t")
        df_rows.append(
            {
                "iteration": int(cells[0]),
                "agent": int(cells[1]),
                "performance": float(cells[2]),
                "private_performance": float(cells[3]),
                "power": float(cells[4]),
                "heat": float(cells[5]),
                "gas": float(cells[6]),
                "p2h": float(cells[7]),
                "p2g": float(cells[8]),
            }
        )
    pd.DataFrame(df_rows).to_csv(f"{filename}_history.csv")

    return {"ziel": soll, "ist": ist, "time": time.time() - global_start_time}


async def wait_for_term(agents):
    def get_negotiation_termination_controller_role():
        for agent in agents:
            for role in agent.roles:
                if type(role) is NegotiationTerminationRole:
                    if role._is_controller:
                        return role

    negotiation_termination_controller_role = (
        get_negotiation_termination_controller_role()
    )
    await asyncio.sleep(0.1)
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
    solar = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.0001874580058407,
        0.0015270868975554,
        0.0026769964211354,
        0.0038257745591754,
        0.0061418349571946,
        0.0155419124925599,
        0.0237460278568065,
        0.031964643380709,
        0.0429912213760122,
        0.0715450950470349,
        0.0971256513634032,
        0.122697148096535,
        0.150560112092094,
        0.1925885579647,
        0.232222917218015,
        0.271852182773238,
        0.314806732042316,
        0.375511516312298,
        0.433441435829143,
        0.491382331775775,
        0.547338878050542,
        0.596264492107872,
        0.645968936469177,
        0.695665180059844,
        0.741288752163066,
        0.775538313063295,
        0.810597852509958,
        0.845658174797026,
        0.87660546038796,
        0.893178679531221,
        0.911289500183215,
        0.929394495297822,
        0.9375955255487,
        0.887934082343557,
        0.847787743267431,
        0.807636812553217,
        0.767222329360787,
        0.710587195620888,
        0.657877207477716,
        0.605176045125917,
        0.552662913274611,
        0.494321271575053,
        0.437541931325793,
        0.380761578344725,
        0.326899847542852,
        0.28036173039361,
        0.233445650911756,
        0.186543663694167,
        0.144005325120327,
        0.115422362911761,
        0.0855336631206063,
        0.0556500724410242,
        0.0300357429549431,
        0.0238133529854286,
        0.0148885833587954,
        0.0059516342467003,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    schedule = get_schedule(np.array(solar), count, power, cut)
    return schedule


def get_schedule(day, count, power, cut):
    schedule = []
    for _ in range(count):
        schedule.append(list(day * power * (1 - np.random.random(96) * cut)))
    return schedule


POWER_TARGET_I = (
    [
        1500,
        2500,
        1500,
        1500,
        1500,
        1500,
        1500,
        500,
        800,
        800,
        800,
        900,
        900,
        900,
        500,
        500,
        200,
        100,
        800,
        410,
        1100,
        600,
        650,
        700,
        800,
        900,
        1000,
        1100,
        1200,
        1300,
        1400,
        2500,
        3600,
        2700,
        2700,
        3700,
        4700,
        3650,
        2600,
        1550,
        1500,
        1450,
        1400,
        1350,
        1300,
        1300,
        1300,
        1500,
        1700,
        1900,
        3100,
        4300,
        4500,
        3500,
        4500,
        4450,
        3400,
        3350,
        2300,
        2250,
        2200,
        2150,
        2100,
        2050,
        2000,
        1950,
        1900,
        1850,
        1850,
        1850,
        1850,
        1900,
        2000,
        2100,
        2200,
        3000,
        4500,
        4300,
        3000,
        3900,
        4900,
        4900,
        4900,
        4900,
        4900,
        4900,
        3900,
        2900,
        1800,
        1700,
        1500,
        1300,
        1100,
        900,
        700,
        500,
    ],
)
HEAT_TARGET_I = (
    [
        1000,
        1000,
        1000,
        1000,
        1200,
        1800,
        2000,
        2500,
        2500,
        2500,
        2500,
        2950,
        2900,
        2850,
        2800,
        2750,
        2700,
        2650,
        2600,
        2550,
        2500,
        2450,
        2400,
        2350,
        2300,
        2250,
        2200,
        2150,
        2100,
        2050,
        2000,
        1950,
        1700,
        1700,
        1700,
        1700,
        1700,
        1800,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        1900,
        2000,
        2100,
        2400,
        2600,
        2850,
        2900,
        2700,
        2800,
        2700,
        2600,
        2100,
        1800,
        1800,
        1800,
        1800,
        1800,
        1850,
        1800,
        1850,
        1700,
        1550,
        1500,
        1550,
        1500,
        1550,
        1700,
        1850,
        1500,
        1550,
        1500,
        1550,
        1400,
        1350,
        1300,
        1350,
        1300,
        1250,
        1100,
    ],
)


def case_industry(run_id, postfix, use_classic=False):
    max_iterations = 2
    max_iteration_power = 0
    penalty_exponent = DEFAULT_PENALTY_EXP
    power_penalty = DEFAULT_PENALTY
    heat_penalty = DEFAULT_PENALTY
    power_kwh_price = 0.15
    heat_kwh_price = 0.1
    converted_price = 0.05
    maximum_agent_attempts = 5
    # 'convert_amount', 'gas_price', 'max_gas_amount', 'gas_to_heat_factor', 'gas_to_power_factor', 'power_to_heat_factor', 'power_to_heat_amount'
    val = [
        [1500, 999 * 0.11 / 430, 1250, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [1500, 999 * 0.11 / 430, 1250, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [1500, 999 * 0.11 / 430, 1250, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [1500, 999 * 0.11 / 430, 1250, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
        [800, 999 * 0.11 / 430, 705, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
        [900, 999 * 0.11 / 430, 710, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
    ]

    power_target = POWER_TARGET_I[0] + np.ones(96) * 500
    heat_target = HEAT_TARGET_I[0] + np.ones(96) * 500

    value_weights = []
    schedules_provider = []
    for v in val:
        value_weights.append(
            {
                "convert_amount": v[0],
                "gas_price": v[1],
                "max_gas_amount": v[2],
                "gas_to_heat_factor": v[3],
                "gas_to_power_factor": v[4],
                "power_to_heat_factor": v[5],
                "power_to_heat_amount": v[6],
                "power_penalty": power_penalty,
                "heat_penalty": heat_penalty,
                "power_kwh_price": power_kwh_price,
                "heat_kwh_price": heat_kwh_price,
                "converted_price": converted_price,
                "penalty_exponent": penalty_exponent,
                "max_iterations": max_iterations,
                "maximum_agent_attempts": maximum_agent_attempts,
                "max_iteration_power": max_iteration_power,
                "name": v[-1],
            }
        )
        if v[7] == 0:
            schedules_provider.append([np.zeros(96).tolist()])
        else:
            schedules_provider.append(v[7])

    asyncio.run(
        test_case(
            power_target=power_target,
            heat_target=heat_target,
            value_weights=value_weights,
            schedules_provider=schedules_provider,
            storages=[],
            name=run_id + "/industry" + postfix,
            use_classic=use_classic
        )
    )


def case_storage_improvement(run_id, penalty_exp=DEFAULT_PENALTY_EXP, additional_name=""):
    max_iterations = 2
    max_iteration_power = 0
    penalty_exponent = penalty_exp
    power_penalty = DEFAULT_PENALTY
    heat_penalty = DEFAULT_PENALTY
    power_kwh_price = 0.15
    heat_kwh_price = 0.1
    converted_price = 0.05
    maximum_agent_attempts = 5
    # 'convert_amount', 'gas_price', 'max_gas_amount', 'gas_to_heat_factor', 'gas_to_power_factor', 'power_to_heat_factor', 'power_to_heat_amount'
    val = [
        [1500, 999 * 0.11 / 430, 1250, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [1500, 999 * 0.11 / 430, 1250, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [1500, 999 * 0.11 / 430, 1250, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [1500, 999 * 0.11 / 430, 1250, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
        [800, 999 * 0.11 / 430, 705, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
        [900, 999 * 0.11 / 430, 710, 951 / 430, 999 / 430, 0, 0, 0, "CHP"],
    ]
    # sector, max_p_bat, min_p_bat, capacity, eff_charge, eff_discharge, start_soc, end_min_soc (all in W/Wh)
    storages = [
        ["heat", 1500, -1500, 12000, 0.99, 0.99, 0.5, 0],
        ["power", 1500, -1500, 12000, 0.95, 0.95, 0.5, 0],
        ["heat", 1500, -1500, 16000, 0.99, 0.99, 0.5, 0],
        ["power", 1500, -1500, 16000, 0.95, 0.95, 0.5, 0],
    ]

    power_target = POWER_TARGET_I[0] + np.ones(96) * 500
    heat_target = HEAT_TARGET_I[0] + np.ones(96) * 500

    value_weights = []
    schedules_provider = []
    for v in val:
        value_weights.append(
            {
                "convert_amount": v[0],
                "gas_price": v[1],
                "max_gas_amount": v[2],
                "gas_to_heat_factor": v[3],
                "gas_to_power_factor": v[4],
                "power_to_heat_factor": v[5],
                "power_to_heat_amount": v[6],
                "power_penalty": power_penalty,
                "heat_penalty": heat_penalty,
                "power_kwh_price": power_kwh_price,
                "heat_kwh_price": heat_kwh_price,
                "converted_price": converted_price,
                "penalty_exponent": penalty_exponent,
                "max_iterations": max_iterations,
                "maximum_agent_attempts": maximum_agent_attempts,
                "max_iteration_power": max_iteration_power,
                "name": v[-1],
            }
        )
        if v[7] == 0:
            schedules_provider.append([np.zeros(96).tolist()])
        else:
            schedules_provider.append(v[7])

    asyncio.run(
        test_case(
            power_target=power_target,
            heat_target=heat_target,
            value_weights=value_weights,
            schedules_provider=schedules_provider,
            storages=storages,
            name=run_id + f"/storage{additional_name}",
        )
    )


def case_storage_improvement_opp(run_id):
    max_iterations = 2
    max_iteration_power = 0
    penalty_exponent = LOW_PENALTY_EXP
    power_penalty = DEFAULT_PENALTY
    heat_penalty = DEFAULT_PENALTY
    power_kwh_price = 0.13
    heat_kwh_price = 0.1
    converted_price = 0.05
    maximum_agent_attempts = 5
    opportunity_list = read_smard("experiments/15_03_2023_gh_15min.csv")
    # 'convert_amount', 'gas_price', 'max_gas_amount', 'gas_to_heat_factor', 'gas_to_power_factor', 'power_to_heat_factor', 'power_to_heat_amount'
    val = [
        [
            1500,
            999 * 0.11 / 430,
            1250,
            951 / 430,
            999 / 430,
            0,
            0,
            0,
            "CHP",
            opportunity_list,
        ],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [
            1500,
            999 * 0.11 / 430,
            1250,
            951 / 430,
            999 / 430,
            0,
            0,
            0,
            "CHP",
            opportunity_list,
        ],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [
            1500,
            999 * 0.11 / 430,
            1250,
            951 / 430,
            999 / 430,
            0,
            0,
            0,
            "CHP",
            opportunity_list,
        ],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [12, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [
            1500,
            999 * 0.11 / 430,
            1250,
            951 / 430,
            999 / 430,
            0,
            0,
            0,
            "CHP",
            opportunity_list,
        ],
        [
            800,
            999 * 0.11 / 430,
            705,
            951 / 430,
            999 / 430,
            0,
            0,
            0,
            "CHP",
            opportunity_list,
        ],
        [
            900,
            999 * 0.11 / 430,
            710,
            951 / 430,
            999 / 430,
            0,
            0,
            0,
            "CHP",
            opportunity_list,
        ],
    ]
    # sector, max_p_bat, min_p_bat, capacity, eff_charge, eff_discharge, start_soc, end_min_soc (all in W/Wh)
    storages = [
        ["heat", 1500, -1500, 12000, 0.99, 0.99, 0.5, 0, opportunity_list],
        ["power", 1500, -1500, 12000, 0.95, 0.95, 0.5, 0, opportunity_list],
        ["heat", 1500, -1500, 16000, 0.99, 0.99, 0.5, 0, opportunity_list],
        ["power", 1500, -1500, 16000, 0.95, 0.95, 0.5, 0, opportunity_list],
    ]

    power_target = POWER_TARGET_I[0] + np.ones(96) * 500
    heat_target = HEAT_TARGET_I[0] + np.ones(96) * 500

    value_weights = []
    schedules_provider = []
    for v in val:
        value_weights.append(
            {
                "convert_amount": v[0],
                "gas_price": v[1],
                "max_gas_amount": v[2],
                "gas_to_heat_factor": v[3],
                "gas_to_power_factor": v[4],
                "power_to_heat_factor": v[5],
                "power_to_heat_amount": v[6],
                "power_penalty": power_penalty,
                "heat_penalty": heat_penalty,
                "power_kwh_price": power_kwh_price,
                "heat_kwh_price": heat_kwh_price,
                "converted_price": converted_price,
                "penalty_exponent": penalty_exponent,
                "max_iterations": max_iterations,
                "maximum_agent_attempts": maximum_agent_attempts,
                "max_iteration_power": max_iteration_power,
                "name": v[8],
                "opportunity": v[9] if len(v) >= 10 else np.zeros(96),
            }
        )
        if v[7] == 0:
            schedules_provider.append([np.zeros(96).tolist()])
        else:
            schedules_provider.append(v[7])

    asyncio.run(
        test_case(
            power_target=power_target,
            heat_target=heat_target,
            value_weights=value_weights,
            schedules_provider=schedules_provider,
            storages=storages,
            name=run_id + "/st_opp",
        )
    )


def case_electric_with_storage(
    run_id, penalty_exp=DEFAULT_PENALTY_EXP, additional_name=""
):
    max_iterations = 2
    max_iteration_power = 0
    penalty_exponent = penalty_exp
    power_penalty = DEFAULT_PENALTY
    heat_penalty = DEFAULT_PENALTY
    power_kwh_price = 0.15
    heat_kwh_price = 0.1
    converted_price = 0.05
    maximum_agent_attempts = 5
    # 'convert_amount', 'gas_price', 'max_gas_amount', 'gas_to_heat_factor', 'gas_to_power_factor', 'power_to_heat_factor', 'power_to_heat_amount'
    val = [
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 900), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 960), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 760), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_wind_schedule(36, 5, 3.7, 0.1, 0.25), "WIND"],
        [0, 0, 0, 0, 0, 4, 2700, 0, "HEATPUMP"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 880), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 850), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 600), "SOLAR"],
        [0, 0, 0, 0, 0, 4, 2000, 0, "HEATPUMP"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 1100), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 200), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 300), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 400), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_wind_schedule(36, 5, 3.7, 0.1, 0.21), "WIND"],
        [0, 0, 0, 0, 0, 0, 0, get_wind_schedule(36, 5, 3.7, 0.1, 0.18), "WIND"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 600), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 900), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 700), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 900), "SOLAR"],
        [0, 0, 0, 0, 0, 0, 0, get_solar_schedule(5, 0.3, 910), "SOLAR"],
        [0, 0, 0, 0, 0, 4, 1300, 0, "HEATPUMP"],
        [0, 0, 0, 0, 0, 0, 0, get_wind_schedule(36, 5, 3.7, 0.1, 0.19), "WIND"],
        [0, 0, 0, 0, 0, 0, 0, get_wind_schedule(36, 5, 3.7, 0.1, 0.29), "WIND"],
    ]
    # sector, max_p_bat, min_p_bat, capacity, eff_charge, eff_discharge, start_soc, end_min_soc (all in W/Wh)
    storages = [
        ["heat", 10000, -10000, 84000, 0.99, 0.99, 0.6, 0],
        ["power", 10000, -10000, 84000, 0.95, 0.95, 0.6, 0],
    ]

    power_target = POWER_TARGET_I[0] + np.ones(96) * 500
    heat_target = HEAT_TARGET_I[0] + np.ones(96) * 1500

    value_weights = []
    schedules_provider = []
    for v in val:
        value_weights.append(
            {
                "convert_amount": v[0],
                "gas_price": v[1],
                "max_gas_amount": v[2],
                "gas_to_heat_factor": v[3],
                "gas_to_power_factor": v[4],
                "power_to_heat_factor": v[5],
                "power_to_heat_amount": v[6],
                "power_penalty": power_penalty,
                "heat_penalty": heat_penalty,
                "power_kwh_price": power_kwh_price,
                "heat_kwh_price": heat_kwh_price,
                "converted_price": converted_price,
                "penalty_exponent": penalty_exponent,
                "max_iterations": max_iterations,
                "maximum_agent_attempts": maximum_agent_attempts,
                "max_iteration_power": max_iteration_power,
                "name": v[-1],
            }
        )
        if v[7] == 0:
            schedules_provider.append([np.zeros(96).tolist()])
        else:
            schedules_provider.append(v[7])

    asyncio.run(
        test_case(
            power_target=power_target,
            heat_target=heat_target,
            value_weights=value_weights,
            schedules_provider=schedules_provider,
            storages=storages,
            name=run_id + f"/electric{additional_name}",
        )
    )


def read_smard(file_name):
    df = pd.read_csv(file_name, delimiter=";")
    return list(
        df["Deutschland/Luxemburg [€/MWh] Berechnete Auflösungen"]
        .apply(lambda v: v.replace(",", "."))
        .astype(float)
        * 0.001
    )[:96]


def case_electric_with_storage_multi_purpose(run_id):
    max_iterations = 2
    max_iteration_power = 0
    penalty_exponent = LOW_PENALTY_EXP
    power_penalty = DEFAULT_PENALTY
    heat_penalty = DEFAULT_PENALTY
    power_kwh_price = 0.14
    heat_kwh_price = 0.1
    converted_price = 0.05
    maximum_agent_attempts = 5
    opportunity_list = read_smard("experiments/15_03_2023_gh_15min.csv")
    # 'convert_amount', 'gas_price', 'max_gas_amount', 'gas_to_heat_factor', 'gas_to_power_factor', 'power_to_heat_factor', 'power_to_heat_amount'
    val = [
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 900),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 960),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 760),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_wind_schedule(36, 5, 3.7, 0.1, 0.25),
            "WIND",
        ],
        [0, 0, 0, 0, 0, 4, 2700, 0, "HEATPUMP", opportunity_list],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 880),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 850),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 600),
            "SOLAR",
        ],
        [0, 0, 0, 0, 0, 4, 2000, 0, "HEATPUMP", opportunity_list],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 1100),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 200),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 300),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 400),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_wind_schedule(36, 5, 3.7, 0.1, 0.21),
            "WIND",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_wind_schedule(36, 5, 3.7, 0.1, 0.18),
            "WIND",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 600),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 900),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 700),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 900),
            "SOLAR",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_solar_schedule(5, 0.3, 910),
            "SOLAR",
        ],
        [0, 0, 0, 0, 0, 4, 1300, 0, "HEATPUMP", opportunity_list],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_wind_schedule(36, 5, 3.7, 0.1, 0.19),
            "WIND",
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            get_wind_schedule(36, 5, 3.7, 0.1, 0.29),
            "WIND",
        ],
    ]
    # sector, max_p_bat, min_p_bat, capacity, eff_charge, eff_discharge, start_soc, end_min_soc (all in W/Wh)
    storages = [
        ["heat", 10000, -10000, 84000, 0.99, 0.99, 0.6, 0, opportunity_list],
        ["power", 10000, -10000, 84000, 0.95, 0.95, 0.6, 0, opportunity_list],
    ]

    power_target = POWER_TARGET_I[0] + np.ones(96) * 500
    heat_target = HEAT_TARGET_I[0] + np.ones(96) * 1500

    value_weights = []
    schedules_provider = []
    for v in val:
        value_weights.append(
            {
                "convert_amount": v[0],
                "gas_price": v[1],
                "max_gas_amount": v[2],
                "gas_to_heat_factor": v[3],
                "gas_to_power_factor": v[4],
                "power_to_heat_factor": v[5],
                "power_to_heat_amount": v[6],
                "power_penalty": power_penalty,
                "heat_penalty": heat_penalty,
                "power_kwh_price": power_kwh_price,
                "heat_kwh_price": heat_kwh_price,
                "converted_price": converted_price,
                "penalty_exponent": penalty_exponent,
                "max_iterations": max_iterations,
                "maximum_agent_attempts": maximum_agent_attempts,
                "max_iteration_power": max_iteration_power,
                "name": v[8],
                "opportunity": v[9] if len(v) >= 10 else np.zeros(96),
            }
        )
        if v[7] == 0:
            schedules_provider.append([np.zeros(96).tolist()])
        else:
            schedules_provider.append(v[7])

    asyncio.run(
        test_case(
            power_target=power_target,
            heat_target=heat_target,
            value_weights=value_weights,
            schedules_provider=schedules_provider,
            storages=storages,
            name=run_id + "/el_plus_opp",
        )
    )


if __name__ == "__main__":
    run_id = str(uuid.uuid1())

    os.makedirs(f"log/{run_id}")
    """
    case_storage_improvement(run_id, LOW_PENALTY_EXP, "_low_penalty")

    cohda.print_data = {}
    reset_globals()

    case_electric_with_storage(run_id, LOW_PENALTY_EXP, "_low_penalty")

    cohda.print_data = {}
    reset_globals()

    case_storage_improvement_opp(run_id)

    cohda.print_data = {}
    reset_globals()

    case_electric_with_storage_multi_purpose(run_id)

    cohda.print_data = {}
    reset_globals()

    case_electric_with_storage(run_id)

    cohda.print_data = {}
    reset_globals()

    case_storage_improvement(run_id)

    cohda.print_data = {}
    reset_globals()

    case_industry(run_id)
    """
    cohda.print_data = {}
    reset_globals()

    case_industry(run_id, "_sampling", use_classic=True)
