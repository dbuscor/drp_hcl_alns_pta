use crate::instance_data::InstanceData;
use cached::proc_macro::cached;
use itertools::Itertools;
use rand::Rng;

#[cached(key = "(u16, u16)", convert = r#"{ (i, j) }"#)]
pub fn distance(i: u16, j: u16, instance: &InstanceData) -> f32 {
    let i_x = &instance.nodes[&i].x;
    let i_y = &instance.nodes[&i].y;
    let j_x = &instance.nodes[&j].x;
    let j_y = &instance.nodes[&j].y;
    ((*i_x - *j_x) * (*i_x - *j_x) + (*i_y - *j_y) * (*i_y - *j_y)).sqrt()
}

pub fn distance_pq(p_1: f32, p_2: f32, q_1: f32, q_2: f32) -> f32 {
    ((q_1 - p_1) * (q_1 - p_1) + (q_2 - p_2) * (q_2 - p_2)).sqrt()
}

#[cached(
    key = "(Vec<u16>, String)",
    convert = r#"{ (path.to_vec(), truck_type.to_string()) }"#
)]
pub fn check_transitions(path: &[u16], truck_type: &str, instance: &InstanceData) -> bool {
    let start_transition = instance.truck_types[truck_type]
        .transitions
        .iter()
        .filter(|&t| t.2 == "start")
        .map(|t| t.1)
        .collect::<Vec<u16>>();
    assert_eq!(start_transition.len(), 1);
    let mut states = vec![start_transition[0]];
    for stop in &path[1..] {
        let new_state = instance.truck_types[truck_type]
            .transitions
            .iter()
            .filter(|&t| t.0 == states[states.len() - 1] && t.2 == instance.nodes[stop].r)
            .map(|t| t.1)
            .collect::<Vec<u16>>();
        assert!(new_state.len() <= 1);
        if new_state.len() == 0 {
            return false;
        }
        states.push(new_state[0]);
    }
    true
}

pub fn check_tw(path: &[u16], start_time: f32, instance: &InstanceData) -> bool {
    let mut start_of_service: Vec<f32> = vec![0.0; path.len()];
    start_of_service[0] = start_time;
    for i in 1..path.len() {
        let start_of_service_i = instance.nodes[&path[i]].tw_e.max(
            start_of_service[i - 1]
                + instance.nodes[&path[i - 1]].st
                + distance(path[i - 1], path[i], instance),
        );
        if start_of_service_i > instance.nodes[&path[i]].tw_l {
            return false;
        }
        start_of_service[i] = start_of_service_i;
    }
    true
}

pub fn distance_between_requests(i: u16, j: u16, instance: &InstanceData) -> f32 {
    assert!(instance.pickup_ids.contains(&i) && instance.pickup_ids.contains(&j));
    distance(i, j, instance)
        + distance(
            instance.delivery_of_pickup[&i],
            instance.delivery_of_pickup[&j],
            instance,
        )
}

pub fn visit_time(i: u16, i_minus_1: u16, time_at_i_minus_1: f32, instance: &InstanceData) -> f32 {
    if i == 0 {
        instance.nodes[&instance.depot_ids[0]].tw_e
    } else {
        instance.nodes[&i].tw_e.max(
            time_at_i_minus_1 + instance.nodes[&i_minus_1].st + distance(i_minus_1, i, instance),
        )
    }
}

pub fn calculate_index<T: Rng>(p: u8, len: usize, rng: &mut T) -> usize {
    let mut f_index: f32 = rng.gen_range(0.0..1.0);
    f_index = (f_index.powi(p as i32)) * (len as f32);
    f_index.floor() as usize
}

pub fn get_compatibility(f: &str, request_collection: &[u16], instance: &InstanceData) -> usize {
    request_collection
        .iter()
        .filter(|&u| {
            instance.truck_types[f]
                .compatible_containers
                .contains(&instance.request_container[u])
        })
        .count()
}

pub fn get_capacity(f: &str, request_collection: &[u16], instance: &InstanceData) -> usize {
    let result: u8 = request_collection
        .iter()
        .map(|r| &instance.request_container[r])
        .unique()
        .map(|c| {
            if instance.truck_types[f].capacities.contains_key(c) {
                instance.truck_types[f].capacities[c]
            } else {
                0
            }
        })
        .sum();
    result as usize
}
