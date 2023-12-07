use crate::utils::distance_pq;
use itertools::Itertools;
use regex::Regex;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use substring::Substring;

#[derive(Debug, Clone)]
pub struct Node {
    pub x: f32,
    pub y: f32,
    pub r: String,
    pub tw_e: f32,
    pub st: f32,
    pub tw_l: f32,
}

#[derive(Debug, Clone)]
pub struct TruckInfo {
    pub fixed_cost: f32,
    pub routing_cost: f32,
    pub transitions: Vec<(u16, u16, String)>,
    pub compatible_containers: HashSet<String>,
    pub capacities: HashMap<String, u8>,
}

#[derive(Debug, Clone)]
pub struct InstanceData {
    pub name: String,
    pub nodes: HashMap<u16, Node>,
    pub depot_ids: Vec<u16>,
    pub pickup_ids: Vec<u16>,
    pub delivery_ids: Vec<u16>,
    pub delivery_of_pickup: HashMap<u16, u16>,
    pub pickup_of_delivery: HashMap<u16, u16>,
    pub truck_types: HashMap<String, TruckInfo>,
    pub request_container: HashMap<u16, String>,
    pub compatible_requests: Vec<Vec<u16>>,
}

impl InstanceData {
    pub fn new(
        name: String,
        id_list: Vec<u16>,
        pd_pairs: Vec<(u16, u16)>,
        depot_ids: Vec<u16>,
        x_list: Vec<f32>,
        y_list: Vec<f32>,
        r_list: Vec<String>,
        e_list: Vec<f32>,
        s_list: Vec<f32>,
        l_list: Vec<f32>,
        truck_types: Vec<String>,
        trucks_fixed_costs: Vec<f32>,
        trucks_routing_costs: Vec<f32>,
        trucks_transitions: Vec<Vec<(u16, u16, String)>>,
    ) -> Self {
        let lists_lengths = [
            id_list.len(),
            x_list.len(),
            y_list.len(),
            r_list.len(),
            e_list.len(),
            s_list.len(),
            l_list.len(),
        ];
        let truck_lists_lengths = [
            truck_types.len(),
            trucks_fixed_costs.len(),
            trucks_routing_costs.len(),
            trucks_transitions.len(),
        ];

        let all_same_length =
            (1..lists_lengths.len()).all(|i| lists_lengths[i] == lists_lengths[0]);
        let all_same_length_trucks = (1..truck_lists_lengths.len())
            .all(|i| truck_lists_lengths[i] == truck_lists_lengths[0]);
        assert!(all_same_length && all_same_length_trucks);

        let mut ns: HashMap<u16, Node> = HashMap::with_capacity(id_list.len());
        for i in 0..id_list.len() {
            ns.insert(
                id_list[i],
                Node {
                    x: x_list[i],
                    y: y_list[i],
                    r: r_list[i].clone(),
                    tw_e: e_list[i],
                    st: s_list[i],
                    tw_l: l_list[i],
                },
            );
        }

        let re = Regex::new(r",(.+?)\)").unwrap();
        let trim_string = |s: &str| -> String {
            let trimmed_s = s.substring(1, s.len() - 1);
            trimmed_s.to_string()
        };

        let mut tt: HashMap<String, TruckInfo> = HashMap::with_capacity(truck_types.len());
        for i in 0..truck_types.len() {
            let i_compatible_containers =
                InstanceData::get_compatible_containers(&trucks_transitions[i]);
            let i_capacities = InstanceData::get_capacities(&trucks_transitions[i]);

            let capacities_containers = i_capacities
                .iter()
                .map(|(k, _v)| k.clone())
                .collect::<HashSet<String>>();
            if i_compatible_containers != capacities_containers {
                panic!("compatible containers and capacities do not match!")
            }

            tt.insert(
                truck_types[i].clone(),
                TruckInfo {
                    fixed_cost: trucks_fixed_costs[i],
                    routing_cost: trucks_routing_costs[i],
                    transitions: trucks_transitions[i].clone(),
                    compatible_containers: i_compatible_containers,
                    capacities: i_capacities,
                },
            );
        }

        let mut pickups: Vec<u16> = Vec::with_capacity(pd_pairs.len());
        let mut deliveries: Vec<u16> = Vec::with_capacity(pd_pairs.len());
        let mut d_of_p: HashMap<u16, u16> = HashMap::with_capacity(pd_pairs.len());
        let mut p_of_d: HashMap<u16, u16> = HashMap::with_capacity(pd_pairs.len());
        for &(p, d) in pd_pairs.iter() {
            pickups.push(p);
            deliveries.push(d);
            d_of_p.insert(p, d);
            p_of_d.insert(d, p);
        }

        let get_container_type = |p: &u16| -> String {
            let r = &ns[p].r;
            trim_string(re.find(r).unwrap().as_str())
        };

        let mut rc: HashMap<u16, String> = HashMap::with_capacity(pickups.len());
        for i in 0..pickups.len() {
            rc.insert(pickups[i], get_container_type(&pickups[i]));
        }

        let er = tt
            .iter()
            .map(|(_, t_info)| {
                t_info
                    .compatible_containers
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<BTreeSet<String>>()
            })
            .unique()
            .collect::<Vec<BTreeSet<String>>>()
            .iter()
            .map(|c_set| {
                rc.iter()
                    .filter(|&(_, container)| (*c_set).contains(container))
                    .map(|(r, _)| *r)
                    .sorted_unstable()
                    .collect::<Vec<u16>>()
            })
            .filter(|c_set| c_set.len() > 1)
            .collect::<Vec<Vec<u16>>>();

        Self {
            name: name.clone(),
            nodes: ns,
            depot_ids,
            pickup_ids: pickups,
            delivery_ids: deliveries,
            delivery_of_pickup: d_of_p,
            pickup_of_delivery: p_of_d,
            truck_types: tt,
            request_container: rc,
            compatible_requests: er,
        }
    }

    fn trim_string(s: &str) -> String {
        let trimmed_s = s.substring(1, s.len() - 1);
        trimmed_s.to_string()
    }

    fn get_compatible_containers(t_transitions: &[(u16, u16, String)]) -> HashSet<String> {
        let re = Regex::new(r",(.+?)\)").unwrap();
        let t_compatible_containers = (*t_transitions)
            .iter()
            .filter(|&x| x.2 != "start" && x.2 != "end")
            .map(|x| InstanceData::trim_string(re.find(&x.2).unwrap().as_str()))
            .unique()
            .collect::<HashSet<String>>();
        t_compatible_containers
    }

    fn get_capacities(t_transitions: &[(u16, u16, String)]) -> HashMap<String, u8> {
        let re = Regex::new(r",(.+?)\)").unwrap();
        let states = t_transitions
            .iter()
            .filter(|&x| x.2 == "start")
            .map(|x| x.1)
            .collect::<Vec<u16>>();
        if states.len() != 1 {
            panic!("There is a problem with the start state!");
        }
        let empty_state = states[0];
        let mut pickups_map = t_transitions
            .iter()
            .filter(|&x| x.0 == empty_state && x.2.contains("(+,"))
            .map(|x| {
                (
                    InstanceData::trim_string(re.find(&x.2).unwrap().as_str()),
                    vec![x.to_owned()],
                )
            })
            .collect::<HashMap<String, Vec<(u16, u16, String)>>>();

        for a in pickups_map.values_mut() {
            loop {
                let next_t = t_transitions
                    .iter()
                    .filter(|&x| a[a.len() - 1].2 == x.2 && a[a.len() - 1].1 == x.0)
                    .map(|x| x.to_owned())
                    .collect::<Vec<(u16, u16, String)>>();
                if next_t.is_empty() {
                    break;
                }
                if next_t.len() > 1 {
                    panic!("There is a problem with the next state!");
                }
                a.push(next_t[0].clone());
            }
        }

        let capacities = pickups_map
            .iter()
            .map(|(k, v)| (k.clone(), v.len() as u8))
            .collect::<HashMap<String, u8>>();
        capacities
    }

    pub fn get_longest_distance(&self) -> f32 {
        let min_x = self
            .nodes
            .iter()
            .map(|(_, n)| n.x)
            .reduce(f32::min)
            .unwrap();
        let min_y = self
            .nodes
            .iter()
            .map(|(_, n)| n.y)
            .reduce(f32::min)
            .unwrap();
        let max_x = self
            .nodes
            .iter()
            .map(|(_, n)| n.x)
            .reduce(f32::max)
            .unwrap();
        let max_y = self
            .nodes
            .iter()
            .map(|(_, n)| n.y)
            .reduce(f32::max)
            .unwrap();
        distance_pq(min_x, min_y, max_x, max_y)
    }

    pub fn get_planning_horizon_length(&self) -> f32 {
        self.nodes[&self.depot_ids[0]].tw_l - self.nodes[&self.depot_ids[0]].tw_e
    }
}

pub fn read_instance(json_path: &str) -> Result<InstanceData, std::io::Error> {
    // Read the input file to string.
    let contents = fs::read_to_string(json_path).expect("Unable to read file");
    // Deserialize and print Rust data structure.
    let data: Value = serde_json::from_str(&contents)?;
    let name = data["name"].as_str().unwrap();
    let mut id_list: Vec<u16> = Vec::new();
    let mut depot_ids: Vec<u16> = Vec::new();
    let mut x_list: Vec<f32> = Vec::new();
    let mut y_list: Vec<f32> = Vec::new();
    let mut r_list: Vec<String> = Vec::new();
    let mut e_list: Vec<f32> = Vec::new();
    let mut s_list: Vec<f32> = Vec::new();
    let mut l_list: Vec<f32> = Vec::new();
    let mut truck_types: Vec<String> = Vec::new();
    let mut trucks_fixed_costs: Vec<f32> = Vec::new();
    let mut trucks_routing_costs: Vec<f32> = Vec::new();
    let mut trucks_transitions: Vec<Vec<(u16, u16, String)>> = Vec::new();

    let mut set_node = |value: &Value| {
        let id = &value["id"].as_u64().unwrap();
        let x = &value["x"].as_f64().unwrap();
        let y = &value["y"].as_f64().unwrap();
        let r = value["r"].as_str().unwrap();
        let e = &value["e"].as_f64().unwrap();
        let s = &value["s"].as_f64().unwrap();
        let l = &value["l"].as_f64().unwrap();
        id_list.push(*id as u16);
        x_list.push(*x as f32);
        y_list.push(*y as f32);
        r_list.push(r.to_string());
        e_list.push(*e as f32);
        s_list.push(*s as f32);
        l_list.push(*l as f32);
    };

    let start_depot = &data["nodes"]["depot"][0];
    set_node(start_depot);

    let pickups = data["nodes"]["pickup"].as_array().unwrap();
    for pickup in pickups.iter() {
        set_node(pickup);
    }

    let deliveries = data["nodes"]["delivery"].as_array().unwrap();
    for delivery in deliveries.iter() {
        set_node(delivery);
    }
    let end_depot = &data["nodes"]["depot"][1];
    set_node(end_depot);

    depot_ids.push(start_depot["id"].as_u64().unwrap() as u16);
    depot_ids.push(end_depot["id"].as_u64().unwrap() as u16);
    assert_eq!(pickups.len(), deliveries.len());
    let mut pd_pairs: Vec<(u16, u16)> = vec![(u16::MAX, u16::MAX); pickups.len()];
    for i in 0..pickups.len() {
        let pickup = &pickups[i]["id"].as_u64().unwrap();
        let delivery = &deliveries[i]["id"].as_u64().unwrap();
        pd_pairs[i].0 = *pickup as u16;
        pd_pairs[i].1 = *delivery as u16;
    }

    let truck_data = data["fleet"].as_array().unwrap();
    for t in truck_data.iter() {
        let t_type = t["truck_type"].as_str().unwrap();
        let t_fixed_cost = t["fixed_cost"].as_f64().unwrap() as f32;
        let t_routing_cost = t["routing_cost"].as_f64().unwrap() as f32;
        let transitions_data = t["transitions"].as_array().unwrap();
        let t_transitions: Vec<(u16, u16, String)> = get_transitions_from_json(transitions_data);

        truck_types.push(t_type.to_string());
        trucks_fixed_costs.push(t_fixed_cost);
        trucks_routing_costs.push(t_routing_cost);
        trucks_transitions.push(t_transitions);
    }

    let instance_data = InstanceData::new(
        name.to_string(),
        id_list,
        pd_pairs,
        depot_ids,
        x_list,
        y_list,
        r_list,
        e_list,
        s_list,
        l_list,
        truck_types,
        trucks_fixed_costs,
        trucks_routing_costs,
        trucks_transitions,
    );
    Ok(instance_data)
}

fn get_transitions_from_json(transitions_data: &[Value]) -> Vec<(u16, u16, String)> {
    transitions_data
        .iter()
        .map(|tt| {
            (
                tt[0].as_u64().unwrap() as u16,
                tt[1].as_u64().unwrap() as u16,
                tt[2].as_str().unwrap().to_string(),
            )
        })
        .collect::<Vec<(u16, u16, String)>>()
}
