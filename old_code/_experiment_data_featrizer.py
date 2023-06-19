from experiment_funcs import *
import numpy as np
import os
import numpy.typing as npt
from typing import Sequence


dataLocation = "data/"
TheodataLocation = "theoretical_data/"
PosdataLocation = "positive_data/"
testdataLocation = 'tests/testDataDir/'
TheoPlayData = 'data_for_dev_try/theoretical/'
ExpPlayData = 'data_for_dev_try/experimental/'
currentLoc = ExpPlayData

query_experimental = True

MPID = "Q0tUKnAE52sy7hVO"

# Posquery_destination = exper_data_query(MPID=MPID, location=PosdataLocation)  #quesries and saves experimental data
current_destination = exper_data_query(MPID=MPID, location=currentLoc, experimental_data = query_experimental)  #quesries and saves theoretical data

Theogoodata_location = exper_data_cleaning(queried_data_string = current_destination, location=currentLoc)  #throws away undesired data, saves the rest in a "goodata"
# Posgoodata_location = exper_data_cleaning(queried_data_string = Posquery_destination, location=PosdataLocation)  #throws away undesired data, saves the rest in a "goodata"
# goodata_location = exper_data_cleaning(queried_data_string = 'tests/testDataDir/testData.npy', location=testdataLocation)  #throws away undesired data, saves the rest in a "goodata"


# # goodata_location = dataLocation+"goodata.npy"
Theogoodata_location = currentLoc+"goodata.npy"
# Posgoodata_location = PosdataLocation+"goodata.npy"

# ###from here analyzing begins


# # a: npt.NDArray[np.complex64]
# def batch_coord(data : Sequence, coord_problem : bool = True) -> tuple[list, np.typing.ArrayLike]:
#     '''Loops through a sequence of data analyzing the coordination environment. 
#     Returns an array, each entity is a dictionary containing the features of one crystal structure.
#     Parameters:
#     ----------------
#     data : Iterable
#         The list/array of crystal data containing Structure and material_id
#     coord_problem : bool
#         Whether to report data points with a problem in analyzing their coordinates. 
#     '''
#     mycoord_problem = []
#     tempList : list[dict] = [{} for _ in range(len(data))]
#     tempArray = np.array(tempList)
#     del tempList

#     for j, datum in enumerate(data):

#         try:
#             oxid_states, sc = analyze_env(struc=datum["structure"], mystrategy="simple")
#             tempArray[j]["oxidation"] = oxid_states
#             tempArray[j]["StructureConn"] = sc
#             tempArray[j]["material_id"] = datum["material_id"]
#             tempArray[j]['e_above_hull'] = datum['e_above_hull']
#             tempArray[j]['formation_energy_per_atom'] = datum['formation_energy_per_atom']

#         except:

#             if coord_problem:

#                 mycoord_problem.append([j, datum["formula"]])

#                 print("Couldn't analyze valencce for ", j, "th datum ",
#                       datum["formula"])

#             else:
#                 print("Couldn't analyze valencce for ", j, "th datum ",
#                       datum["formula"])
#     return mycoord_problem, tempArray





# # goodata = np.load(Posgoodata_location, allow_pickle = True)
# goodata = np.load(Theogoodata_location, allow_pickle = True)
# # goodata = np.load(goodata_location, allow_pickle = True)
# # chunkSize = 10
# chunkSize = 1000

# coord_problem = []
# data_chunk_list = []
# for i, chunk in enumerate(chunker(goodata, chunkSize)):

#     data_chunk_name = "coordata_" + str(i)
#     data_chunk_list.append(data_chunk_name + '.npy')

#     # if os.path.exists(dataLocation + data_chunk_name + '.npy'): ###COMMENT THIS OUT if you want to overwrite previous data!!!!!!!!
#     # if os.path.exists(testdataLocation + data_chunk_name + '.npy'): ###COMMENT THIS OUT if you want to overwrite previous data!!!!!!!!
#     if os.path.exists(currentLoc + data_chunk_name + '.npy'): ###COMMENT THIS OUT if you want to overwrite previous data!!!!!!!!
#     # if os.path.exists(PosdataLocation + data_chunk_name + '.npy'): ###COMMENT THIS OUT if you want to overwrite previous data!!!!!!!!
#         continue

#     print('calculating coordination for the range', i*chunkSize, 'to', min((i+1)*chunkSize, len(goodata)))

#     chunkCoord_problem, tempArray = batch_coord(chunk, coord_problem=True)

#     coord_problem.append(chunkCoord_problem)  #unravel this list later
    
#     # np.save(testdataLocation + data_chunk_name, tempArray)
#     # np.save(dataLocation + data_chunk_name, tempArray)
#     # np.save(PosdataLocation + data_chunk_name, tempArray)
#     np.save(currentLoc + data_chunk_name, tempArray)

#     del tempArray


#     # coord_problem = np.array(coord_problem)
#     # np.save("data/coord_problems_"+str(i), coord_problem)

# print("Files to be featurized are", *data_chunk_list )

# # the separation is there to delete the tempArray and not to overflow the memory

# crystal_features = np.empty(0)  #to initiate appending data below

# for data_chunk_name in data_chunk_list:
#     res= []
#     # sc_data = np.load(testdataLocation + data_chunk_name ,allow_pickle=True)
#     # sc_data = np.load(dataLocation + data_chunk_name ,allow_pickle=True)
#     sc_data = np.load(currentLoc + data_chunk_name ,allow_pickle=True)
#     # sc_data = np.load(PosdataLocation + data_chunk_name ,allow_pickle=True)
#     print('Calculating features for ', data_chunk_name  )
#     for datum in sc_data:
#         try:
#             SC = datum['StructureConn']
#             oxid_states = datum["oxidation"]
#             material_id = datum['material_id']
#             e_above_hull = datum['e_above_hull']
#             formation_energy_per_atom = datum['formation_energy_per_atom']
            
#         except:
#             continue
#         res.append({'feats' :crysFeaturizer(SC,oxid_states), 'e_above_hull': e_above_hull,
#                     'formation_energy_per_atom': formation_energy_per_atom , 'material_id':material_id })
    
#     crystal_features = np.concatenate((crystal_features,np.array(res)))  #collects the features of different chuncks of data.
#     # break


# # np.save(testdataLocation + 'local_crystal_features', crystal_features) 
# # np.save(dataLocation + 'local_crystal_features', crystal_features) 
# # np.save(PosdataLocation + 'local_crystal_features', crystal_features) 
# np.save(currentLoc + 'local_crystal_features', crystal_features) 

