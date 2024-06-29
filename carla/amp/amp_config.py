'''
Objects : 
    - Vehicles 
    - Buses
    - Vans
    - Bicycle
    - Motorcycle
'''

carla_object_type = {
    'Vehicle'               : 0,
    'Buses'                 : 1,
    'Bicycle' 		        : 2,
    'Motorcycle'            : 3,

    'Pedestrian'            : 2,

    'static'                : 6,
    'construction'          : 7,
}

Vehicle = [
    "vehicle.audi.a2" ,
    "vehicle.audi.etron" ,
    "vehicle.audi.tt" ,
    "vehicle.bmw.grandtourer" ,
    "vehicle.chevrolet.impala" ,
    "vehicle.citroen.c3" ,
    "vehicle.dodge.charger_2020" ,
    "vehicle.dodge.charger_police" ,
    "vehicle.dodge.charger_police_2020" ,
    "vehicle.ford.crown" , 
    "vehicle.ford.mustang" ,
    "vehicle.jeep.wrangler_rubicon" ,
    "vehicle.lincoln.mkz_2020" ,
    "vehicle.mercedes.coupe" ,
    "vehicle.mercedes.coupe_2020" ,
    "vehicle.micro.microlino" ,
    "vehicle.mini.cooper_s" ,
    "vehicle.mini.cooper_s_2021" ,
    "vehicle.nissan.micra" ,
    "vehicle.nissan.patrol" ,
    "vehicle.nissan.patrol_2021" ,
    "vehicle.seat.leon" ,
    "vehicle.tesla.model3" ,
    "vehicle.toyota.prius" ,
]


Buses = [
    'vehicle.carlamotors.carlacola' , 
    'vehicle.carlamotors.european_hgv' , 
    'vehicle.carlamotors.firetruck' ,
    'vehicle.tesla.cybertruck' , 
    'vehicle.ford.ambulance' , 
    'vehicle.mercedes.sprinter' ,
    'vehicle.volkswagen.t2' ,
    'vehicle.volkswagen.t2_2021' , 
    'vehicle.mitsubishi.fusorosa' , 
]


Motorcycle = [
    'vehicle.harley-davidson.low_rider' , 
    'vehicle.kawasaki.ninja' ,
    'vehicle.vespa.zx125' ,
    'vehicle.yamaha.yzf'
]

Bicycle = [
    "vehicle.bh.crossbike" , 
    "vehicle.diamondback.century" , 
    "vehicle.gazelle.omafiets" , 

]

Pedestrian = [
    '*walker*'
]

static = [
    'static.prop.clothcontainer' , 
    'static.prop.container' , 
    'static.prop.streetbarrier' , 
    'static.prop.constructioncone' , 
    'static.prop.trafficcone01' , 
    'static.prop.trafficcone02' , 
    'static.prop.warningconstruction' , 
    'static.prop.warningaccident' , 
]


carla_objects_bp = {
    'Vehicle'               : Vehicle,
    'Buses'                 : Buses,
    'Bicycle' 		        : Bicycle,
    'Motorcycle'            : Motorcycle,
    'Pedestrian'            : Pedestrian,
    'static'                : static,
}

LANE_NEAR_DISTANCE=50
OBJECT_TRACK_LENGTH=61
OBJECT_NEAR_DISTANCE=30

CKPT_PATH = '/media/mahmoud/New Volume/faculty/level2/study/machine learning/Towards Enhanced Autonomous Vehicle/carla/amp/VectorNet/Argoverse2/argo-1-best_model.pth'

N_TRAJ=6
N_FUTURE=50
DISPLAY_TRAJ=1