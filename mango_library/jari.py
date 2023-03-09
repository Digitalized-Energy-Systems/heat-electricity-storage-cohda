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
    # logging.basicConfig(level=logging.INFO)
    # logging.info('Läuft bei Dir.')
    # logging.debug('Läuft nicht bei Dir.')
    c = await Container.factory(addr=addr)

    global_start_time = time.time()
    open_value_weights = []
    for value_weight in value_weights:
        value_weight["global_start_time"] = global_start_time
        # TODO Wird es noch gebraucht?
        # open_value_weights.append({'power_kwh_price': value_weight["power_kwh_price"], 'converted_price': value_weight["converted_price"]})

    " create agents "
    agents = []
    addrs = []
    for i, _ in enumerate(schedules_provider):
        a = RoleAgent(c)
        # TODO open_value_weights entfernen
        cohda_role = COHDARole(schedules_provider[i], value_weights[i], open_value_weights, lambda s: True)
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
            filter(lambda a_t: a_t[0] != str(part_id), map(lambda ad: (ad[1], c.addr, ad[0].aid), zip(agents, range(100))))), 'cohda', str(part_id), 'agent0', addr))
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
    # df = pd.DataFrame([[38.0, 2.0, 18.0, 22.0, 21, 0], [19, 439, 6, 452, 226, 232]], columns=pd.MultiIndex.from_product([['Decision Tree', 'Regression', 'Random'], ['Tumour', 'Non-Tumour']], names=["a", "b"]))
    # print(df)
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


""" test """
start = 0
end = 96
# heat_target = [0] * end,
asyncio.run(test_case(
    power_target=[500, 520, 520, 520, 520, 520, 520, 520, 620, 720, 820, 3200, 4200, 5200, 8500, 12520, 8520, 4420, 2520, 1520, 520, 520, 520, 520, 80, 130, 220, 280, 470, 400, 440, 550, 570, 820, 900, 1010, 1230, 1520, 2400, 3000, 3400, 3800, 4200, 4600, 5000, 5400, 4800, 6000, 6100, 6200, 6300, 6400, 6400, 6400, 6500, 6600, 6600, 6500, 6500, 6500, 6400, 6100, 6100, 6200, 6100, 5800, 5600, 5100, 4200, 4600, 4700, 3300, 3300, 1860, 1270, 820, 540, 376, 220, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520, 520][start:end],
    heat_target=[300, 260, 260, 260, 260, 260, 260, 260, 310, 360, 410, 1600, 2100, 2600, 4250, 6260, 4260, 2210, 1260, 760, 260, 260, 260, 260, 40, 65, 110, 140, 235, 200, 220, 275, 285, 410, 450, 505, 615, 760, 1200, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2400, 3000, 3050, 3100, 3150, 3200, 3200, 3200, 3250, 3300, 3300, 3250, 3250, 3250, 3200, 3050, 3050, 3100, 3050, 2900, 2800, 2550, 2100, 2300, 2350, 1650, 1650, 930, 635, 410, 270, 188, 110, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260][start:end],
    value_weights=[
        {'power_penalty': 2, 'heat_penalty': 2, 'power_kwh_price': 1, 'heat_kwh_price': 1, 'converted_price': .8, 'convert_amount': 1000, 'gas_price': 0.07, 'max_gas_amount': 100, 'gas_to_heat_factor': .5, 'gas_to_power_factor': .5, 'power_to_heat_factor': .9, 'power_to_heat_amount': 7},
        {'power_penalty': 2, 'heat_penalty': 2, 'power_kwh_price': 1, 'heat_kwh_price': 1, 'converted_price': .8, 'convert_amount': 1000, 'gas_price': 0.07, 'max_gas_amount': 100, 'gas_to_heat_factor': .5, 'gas_to_power_factor': .5, 'power_to_heat_factor': .9, 'power_to_heat_amount': 7},
        {'power_penalty': 2, 'heat_penalty': 2, 'power_kwh_price': 1, 'heat_kwh_price': 1, 'converted_price': .8, 'convert_amount': 1000, 'gas_price': 0.07, 'max_gas_amount': 100, 'gas_to_heat_factor': .5, 'gas_to_power_factor': .5, 'power_to_heat_factor': .9, 'power_to_heat_amount': 7},
        {'power_penalty': 2, 'heat_penalty': 2, 'power_kwh_price': 1, 'heat_kwh_price': 1, 'converted_price': .8, 'convert_amount': 1000, 'gas_price': 0.07, 'max_gas_amount': 100, 'gas_to_heat_factor': .5, 'gas_to_power_factor': .5, 'power_to_heat_factor': .9, 'power_to_heat_amount': 7},
        {'power_penalty': 2, 'heat_penalty': 2, 'power_kwh_price': 1, 'heat_kwh_price': 1, 'converted_price': .8, 'convert_amount': 1000, 'gas_price': 0.17, 'max_gas_amount': 100, 'gas_to_heat_factor': .5, 'gas_to_power_factor': .5, 'power_to_heat_factor': .9, 'power_to_heat_amount': 7},
    ],
    schedules_provider=[[[100, 104, 112, 111, 112, 107, 104, 110, 135, 157, 166, 688, 880, 1062, 1783, 2506, 1746, 901, 517, 309, 111, 107, 107, 107, 16, 28, 48, 60, 96, 84, 91, 112, 118, 165, 183, 219, 263, 310, 502, 601, 743, 807, 854, 975, 1044, 1135, 1020, 1314, 1331, 1295, 1344, 1354, 1312, 1295, 1396, 1370, 1380, 1388, 1346, 1359, 1299, 1322, 1323, 1316, 1251, 1263, 1149, 1065, 885, 997, 1021, 682, 714, 401, 260, 174, 117, 78, 48, 113, 110, 105, 113, 108, 106, 105, 111, 107, 112, 107, 114, 113, 113, 110, 112, 112][start:end],
                         [101, 104, 105, 112, 110, 112, 108, 113, 127, 155, 176, 669, 908, 1065, 1765, 2684, 1853, 961, 550, 325, 107, 106, 104, 112, 16, 28, 44, 60, 103, 81, 93, 118, 117, 174, 193, 204, 266, 326, 483, 622, 722, 826, 910, 938, 1071, 1120, 1005, 1266, 1244, 1318, 1359, 1378, 1330, 1338, 1340, 1394, 1371, 1389, 1377, 1309, 1360, 1326, 1239, 1348, 1328, 1189, 1180, 1120, 843, 956, 966, 703, 661, 400, 278, 166, 118, 78, 44, 109, 106, 111, 108, 106, 107, 112, 106, 105, 111, 111, 114, 113, 106, 111, 112, 106][start:end]],
                        [[105, 113, 109, 109, 108, 111, 114, 111, 127, 156, 171, 670, 886, 1078, 1770, 2590, 1870, 895, 545, 319, 104, 104, 110, 110, 16, 27, 45, 60, 100, 87, 89, 116, 117, 179, 189, 219, 248, 333, 513, 605, 713, 801, 913, 992, 1038, 1123, 998, 1285, 1296, 1329, 1337, 1360, 1333, 1360, 1427, 1374, 1448, 1326, 1423, 1402, 1369, 1257, 1305, 1340, 1287, 1259, 1148, 1063, 856, 963, 993, 689, 717, 388, 279, 179, 114, 82, 44, 108, 109, 108, 114, 105, 107, 106, 109, 107, 106, 109, 112, 109, 109, 113, 109, 104][start:end],
                         [100, 111, 111, 109, 111, 113, 109, 105, 124, 145, 176, 644, 854, 1132, 1733, 2593, 1839, 962, 542, 315, 104, 107, 109, 104, 17, 26, 45, 57, 102, 83, 89, 110, 119, 170, 194, 208, 268, 311, 506, 637, 701, 834, 922, 1008, 1075, 1184, 1000, 1282, 1273, 1259, 1304, 1381, 1293, 1316, 1367, 1350, 1343, 1399, 1300, 1388, 1329, 1282, 1285, 1298, 1313, 1193, 1230, 1051, 876, 990, 977, 713, 723, 388, 257, 174, 108, 78, 45, 109, 110, 112, 110, 104, 104, 106, 107, 107, 108, 111, 111, 110, 110, 107, 111, 104][start:end]],
                        [[104, 108, 114, 105, 112, 111, 106, 108, 136, 153, 173, 695, 893, 1111, 1710, 2748, 1832, 958, 541, 319, 113, 108, 105, 105, 16, 28, 48, 56, 97, 80, 88, 115, 120, 170, 185, 212, 266, 310, 503, 613, 691, 828, 842, 922, 1058, 1124, 1039, 1298, 1220, 1300, 1347, 1399, 1325, 1360, 1417, 1380, 1365, 1381, 1367, 1399, 1382, 1267, 1230, 1242, 1314, 1249, 1138, 1059, 908, 924, 1018, 715, 692, 395, 269, 171, 113, 79, 45, 112, 109, 109, 104, 113, 112, 106, 114, 104, 106, 109, 110, 111, 107, 109, 112, 113][start:end],
                         [100, 110, 111, 110, 111, 112, 109, 104, 131, 144, 174, 660, 856, 1069, 1809, 2651, 1717, 966, 548, 329, 113, 104, 113, 109, 17, 27, 48, 59, 95, 87, 91, 116, 118, 168, 193, 210, 261, 314, 509, 628, 741, 835, 898, 995, 1081, 1127, 989, 1220, 1325, 1259, 1334, 1313, 1287, 1352, 1320, 1389, 1423, 1379, 1335, 1312, 1284, 1322, 1231, 1245, 1262, 1248, 1175, 1054, 908, 973, 961, 671, 714, 372, 276, 171, 115, 79, 45, 109, 107, 104, 104, 104, 109, 114, 113, 108, 111, 112, 110, 111, 108, 111, 114, 113][start:end]],
                        [[102, 111, 113, 110, 107, 110, 105, 109, 131, 157, 173, 693, 898, 1089, 1750, 2644, 1800, 907, 511, 324, 111, 112, 109, 105, 17, 26, 44, 57, 99, 85, 94, 115, 114, 167, 184, 217, 260, 316, 491, 630, 720, 768, 853, 1000, 1052, 1133, 961, 1294, 1276, 1243, 1284, 1289, 1297, 1389, 1397, 1426, 1385, 1378, 1424, 1415, 1353, 1320, 1238, 1264, 1256, 1195, 1130, 1084, 892, 946, 1011, 712, 687, 399, 265, 174, 114, 75, 46, 109, 113, 104, 108, 105, 107, 113, 108, 106, 111, 110, 105, 112, 105, 107, 104, 113][start:end],
                         [100, 111, 110, 104, 111, 108, 105, 110, 133, 153, 166, 643, 893, 1125, 1844, 2595, 1863, 910, 542, 327, 105, 111, 113, 108, 17, 26, 45, 61, 101, 86, 94, 111, 114, 170, 187, 221, 249, 311, 524, 609, 733, 770, 863, 955, 1018, 1166, 980, 1246, 1321, 1318, 1361, 1347, 1324, 1306, 1323, 1386, 1422, 1420, 1425, 1422, 1372, 1304, 1236, 1294, 1247, 1206, 1207, 1032, 915, 986, 943, 675, 674, 402, 258, 165, 108, 75, 45, 105, 111, 108, 109, 112, 106, 106, 110, 107, 105, 110, 105, 112, 113, 108, 107, 109][start:end]],
                        [[99, 110, 105, 107, 107, 106, 109, 112, 124, 145, 177, 689, 883, 1093, 1727, 2641, 1820, 891, 515, 320, 112, 107, 104, 109, 16, 26, 46, 60, 102, 83, 92, 111, 114, 170, 183, 219, 250, 318, 494, 609, 725, 834, 917, 953, 1073, 1091, 973, 1304, 1296, 1359, 1349, 1323, 1374, 1341, 1381, 1332, 1397, 1354, 1357, 1415, 1374, 1268, 1337, 1317, 1310, 1174, 1177, 1073, 920, 935, 985, 709, 674, 391, 256, 178, 114, 81, 45, 107, 111, 111, 112, 113, 106, 104, 113, 104, 109, 110, 105, 110, 104, 105, 107, 104][start:end],
                         [100, 111, 113, 110, 112, 108, 114, 106, 135, 145, 167, 655, 854, 1080, 1811, 2611, 1833, 953, 551, 331, 111, 107, 112, 109, 16, 26, 47, 57, 97, 84, 96, 120, 125, 177, 182, 210, 259, 324, 522, 635, 726, 827, 868, 1008, 1038, 1081, 1015, 1272, 1262, 1337, 1314, 1385, 1356, 1345, 1364, 1364, 1384, 1405, 1341, 1333, 1349, 1281, 1231, 1270, 1336, 1244, 1193, 1076, 910, 992, 971, 695, 710, 399, 260, 175, 116, 82, 44, 111, 111, 112, 110, 110, 113, 113, 110, 114, 109, 113, 107, 109, 113, 105, 109, 108][start:end]],
                        ],
))

""" test """
# start = 0
# end = 10
# asyncio.run(test_case(
#     power_target=[10, 20, 30, 40, 50, 60, 40, 30, 20, 10][start:end],
#     heat_target=[600, 400, 400, 400, 300, 300, 400, 500, 600, 600][start:end],
#     value_weights=[
#         {'power_penalty': 2, 'heat_penalty': 2, 'gas_price': 0.07, 'power_kwh_price': 1, 'heat_kwh_price': 1, 'converted_price': .8, 'convert_amount': 100, 'max_gas_amount': 1000, 'gas_to_heat_factor': .5, 'gas_to_power_factor': .4, 'power_to_heat_factor': .9, 'power_to_heat_amount': 100},
#         {'power_penalty': 2, 'heat_penalty': 2, 'gas_price': 0.07, 'power_kwh_price': 1, 'heat_kwh_price': 1, 'converted_price': .8, 'convert_amount': 100, 'max_gas_amount': 1000, 'gas_to_heat_factor': .4, 'gas_to_power_factor': .5, 'power_to_heat_factor': .9, 'power_to_heat_amount': 100},
#     ],
#     schedules_provider=
#     [
#         [
#             [10, 20, 30, 40, 50, 60, 40, 30, 20, 10],
#             [10, 20, 30, 40, 50, 60, 40, 30, 20, 10]
#         ],
#         [
#             [10, 20, 30, 40, 50, 60, 40, 30, 20, 10],
#             [10, 20, 30, 40, 50, 60, 40, 30, 20, 10]
#         ]
#     ]
# ))
