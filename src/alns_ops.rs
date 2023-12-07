use crate::instance_data::InstanceData;
use crate::solution::Solution;
use crate::utils::{
    check_transitions, check_tw, distance, distance_between_requests, get_capacity,
    get_compatibility,
};
use cached::proc_macro::cached;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::seq::index::sample_weighted;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
struct InsertionData {
    route_index: Option<usize>,
    pickup_index: Option<usize>,
    delivery_index: Option<usize>,
    insertion_cost: f32,
}

#[derive(Debug)]
struct NewRouteData {
    truck_type: String,
    new_route_cost: f32,
}

#[derive(Debug)]
enum Ins {
    Insertion(InsertionData),
    NewRoute(NewRouteData),
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct DData {
    dissimilarity: OrderedFloat<f32>,
    request_id: u16,
}

fn dissimilarity(
    i: u16,
    j: u16,
    w1: f32,
    w2: f32,
    w3: f32,
    solution: &Solution,
    instance: &InstanceData,
) -> f32 {
    assert!(instance.pickup_ids.contains(&i) && instance.pickup_ids.contains(&j));

    let compatible_truck_types = |request: u16| -> HashSet<&str> {
        let request_truck_types = instance
            .truck_types
            .iter()
            .filter(|&(_, t)| {
                t.compatible_containers
                    .contains(&instance.request_container[&request])
            })
            .map(|(k, _)| k.as_str())
            .collect::<HashSet<&str>>();
        request_truck_types
    };

    let f1 = || -> f32 {
        let dist = (distance(i, j, instance)
            + distance(
                instance.delivery_of_pickup[&i],
                instance.delivery_of_pickup[&j],
                instance,
            ))
            / (2.0 * instance.get_longest_distance());
        dist
    };

    let f2 = || -> f32 {
        let (vt_pickup_i, vt_delivery_i) = solution.get_visit_times(i);
        let (vt_pickup_j, vt_delivery_j) = solution.get_visit_times(j);

        let vt_diff = ((vt_pickup_i - vt_pickup_j).abs() + (vt_delivery_i - vt_delivery_j).abs())
            / (2.0 * instance.get_planning_horizon_length());
        vt_diff
    };

    let f3 = || -> f32 {
        let mut i_compatible_truck_types = compatible_truck_types(i);
        let j_compatible_truck_types = compatible_truck_types(j);
        i_compatible_truck_types.extend(&j_compatible_truck_types);

        let i_req = &instance.request_container[&i];
        let j_req = &instance.request_container[&j];

        let q = |truck_type: &str, request: &str| -> f32 {
            if !instance.truck_types[truck_type]
                .compatible_containers
                .contains(request)
            {
                0.0
            } else {
                instance.truck_types[truck_type].capacities[request] as f32
            }
        };

        let mut result: f32 = 0.0;
        for tt in i_compatible_truck_types.iter() {
            let q_tt_i: f32 = q(*tt, i_req);
            let q_tt_j: f32 = q(*tt, j_req);
            result += (q_tt_i - q_tt_j).abs() / q_tt_i.max(q_tt_j);
        }
        result /= i_compatible_truck_types.len() as f32;
        result
    };

    w1 * f1() + w2 * f2() + w3 * f3()
}

fn no_to_remove(q: f32, l: usize) -> usize {
    (q * (l as f32)).ceil() as usize
}

fn shaw_index<T: Rng>(dissimilarities: &[DData], rng: &mut T) -> usize {
    let total_dissimilarity: f32 = dissimilarities
        .iter()
        .map(|d| d.dissimilarity.into_inner())
        .sum();

    let selected_indexes = sample_weighted(
        rng,
        dissimilarities.len(),
        |i| total_dissimilarity / dissimilarities[i].dissimilarity.into_inner(),
        1,
    )
    .unwrap();
    selected_indexes.into_vec()[0]
}

pub fn shaw_removal<T: Rng>(
    current: &Solution,
    q: f32,
    weights: &[f32; 3],
    rng: &mut T,
    instance: &InstanceData,
) -> Solution {
    let mut solution = current.clone();

    let compatible_requests = instance.compatible_requests.choose(rng).unwrap();
    let r = compatible_requests.choose(rng).unwrap();
    let mut d = vec![*r];

    while d.len() <= no_to_remove(q, compatible_requests.len()) {
        let r_new = d.choose(rng).unwrap();
        let l = compatible_requests
            .iter()
            .filter(|&r| !d.contains(r))
            .map(|r| *r)
            .collect::<Vec<u16>>();

        let mut dissimilarity_vec = l
            .par_iter()
            .map(|x| DData {
                dissimilarity: OrderedFloat(dissimilarity(
                    *r_new, *x, weights[0], weights[1], weights[2], &solution, instance,
                )),
                request_id: *x,
            })
            .collect::<Vec<DData>>();

        let index = shaw_index(&dissimilarity_vec, rng);
        dissimilarity_vec.select_nth_unstable(index);
        let selected_request = &dissimilarity_vec[index].request_id;
        d.push(*selected_request);
    }

    solution.remove_requests(&d, instance);
    solution
}

pub fn random_removal<T: Rng>(
    current: &Solution,
    q: f32,
    rng: &mut T,
    instance: &InstanceData,
) -> Solution {
    let mut solution = current.clone();

    let requests_to_remove = instance
        .pickup_ids
        .choose_multiple(rng, no_to_remove(q, instance.pickup_ids.len()))
        .cloned()
        .collect::<Vec<u16>>();

    solution.remove_requests(&requests_to_remove, instance);
    solution
}

#[cached(
    key = "(u16, Vec<u16>)",
    convert = r#"{ (request, compatible_requests.to_vec()) }"#
)]
pub fn adjacent_requests(
    request: u16,
    compatible_requests: &[u16],
    instance: &InstanceData,
) -> Vec<u16> {
    // request must be the first element of list adjacent_requests(request)
    let mut compatible_req: Vec<u16> = compatible_requests
        .iter()
        .filter(|&r| *r != request)
        .map(|r| *r)
        .collect();
    compatible_req
        .par_sort_by_cached_key(|r| OrderedFloat(distance_between_requests(request, *r, instance)));
    compatible_req.insert(0, request);
    compatible_req
}

fn select_requests_to_remove<T: Rng>(
    r_route: usize,
    l_t: usize,
    r_selected: u16,
    rng: &mut T,
    solution: &Solution,
    compatible_requests: &[u16],
    instance: &InstanceData,
) -> Vec<u16> {
    let requests_that_can_be_removed: Vec<u16> = solution.routes[r_route]
        .stops
        .iter()
        .filter(|&s| {
            !instance.depot_ids.contains(s) && *s != r_selected && compatible_requests.contains(s)
        })
        .cloned()
        .collect();
    // assert!(l_t <= requests_that_can_be_removed.len());

    if requests_that_can_be_removed.len() == 0 {
        // only r_selected can be removed
        return vec![r_selected];
    }

    let distances: Vec<f32> = requests_that_can_be_removed
        .iter()
        .map(|r| distance_between_requests(r_selected, *r, instance))
        .collect();
    let distances_sum: f32 = distances.iter().sum();
    let selected_indexes = sample_weighted(
        rng,
        requests_that_can_be_removed.len(),
        |r| distances_sum / distances[r], // the closer, the more likely to be chosen.
        l_t - 1,                          // r_selected will be added afterward.
    )
    .unwrap();

    let mut requests_to_remove: Vec<u16> = selected_indexes
        .iter()
        .map(|i| requests_that_can_be_removed[i])
        .collect();
    requests_to_remove.push(r_selected);
    assert_eq!(
        (1..requests_to_remove.len())
            .any(|i| requests_to_remove[i..].contains(&requests_to_remove[i - 1])),
        false
    );
    requests_to_remove
}

pub fn sisrs_removal<T: Rng>(
    current: &Solution,
    q: f32,
    rng: &mut T,
    instance: &InstanceData,
) -> Solution {
    let mut solution = current.clone();
    let compatible_requests = instance.compatible_requests.choose(rng).unwrap();
    let route_no_of_requests = |stops: &[u16]| -> usize {
        // compatible_requests contains only pickup ids.
        stops
            .iter()
            .filter(|&s| !instance.depot_ids.contains(s) && (*compatible_requests).contains(s))
            .count()
    };
    let no_requests_of_route: HashMap<usize, usize> = (*compatible_requests)
        .iter()
        .map(|req| solution.request_position[req].route_id)
        .unique()
        .map(|r_id| (r_id, route_no_of_requests(&solution.routes[r_id].stops)))
        .collect();

    let k_s = (q * no_requests_of_route.len() as f32).round() as usize;

    let r_seed_s = (*compatible_requests).choose(rng).unwrap();
    let mut ruined_routes: Vec<usize> = Vec::new();
    let mut requests_to_remove: Vec<u16> = Vec::new();

    for r in adjacent_requests(*r_seed_s, compatible_requests, instance) {
        if ruined_routes.len() < k_s {
            let r_route = solution.request_position[&r].route_id;
            if !requests_to_remove.contains(&r) && !ruined_routes.contains(&r_route) {
                let l_max_t = no_requests_of_route[&r_route] as f32;
                let l_t = rng.gen_range(1.0..(l_max_t + 1.0)).floor() as usize;
                requests_to_remove.append(&mut select_requests_to_remove(
                    r_route,
                    l_t,
                    r,
                    rng,
                    &solution,
                    &compatible_requests,
                    instance,
                ));
                ruined_routes.push(r_route);
            }
        }
    }
    solution.remove_requests(&requests_to_remove, instance);
    solution
}

#[cached(
    key = "(u16, Vec<u16>, String)",
    convert = r#"{ (pickup, stops.to_vec(), truck_type.to_string()) }"#
)]
pub fn calculate_best_insertion(
    pickup: u16,
    stops: &[u16],
    truck_type: &str,
    original_routing_cost: f32,
    instance: &InstanceData,
) -> HashMap<(usize, usize), f32> {
    let truck_info = &instance.truck_types[truck_type];
    let mut insertions: HashMap<(usize, usize), f32> = HashMap::new();
    for i in 1..stops.len() {
        for j in i..stops.len() {
            let mut path_ij = stops.to_vec();
            path_ij.insert(j, instance.delivery_of_pickup[&pickup]);
            path_ij.insert(i, pickup);
            if check_transitions(&path_ij, truck_type, instance)
                && check_tw(&path_ij, instance.nodes[&path_ij[0]].tw_e, instance)
            {
                let updated_routing_cost: f32 = (0..path_ij.len() - 1)
                    .map(|k| {
                        truck_info.routing_cost * distance(path_ij[k], path_ij[k + 1], instance)
                    })
                    .sum();
                let delta_cost_ij = updated_routing_cost - original_routing_cost;
                insertions.insert((i, j), delta_cost_ij);
            }
        }
    }
    insertions
}

pub fn calculate_best_insertion_with_blinks(
    pickup: u16,
    stops: &[u16],
    truck_type: &str,
    original_routing_cost: f32,
    blink_rate: f32,
    instance: &InstanceData,
) -> Option<(f32, usize, usize)> {
    let mut rng = rand::thread_rng();
    let insertions =
        calculate_best_insertion(pickup, stops, truck_type, original_routing_cost, instance);
    if !insertions.is_empty() {
        let mut best_insertion: Option<(f32, usize, usize)> = None;
        for ((i, j), cost) in insertions {
            if rng.gen_range(0.0_f32..1.0) < blink_rate {
                if best_insertion.is_none() || cost < best_insertion.unwrap().0 {
                    best_insertion = Some((cost, i, j));
                }
            }
        }
        best_insertion
    } else {
        None
    }
}

fn identify_compatible_trucks_and_random_select<T: Rng>(
    pickup: &u16,
    rng: &mut T,
    instance: &InstanceData,
) -> (Vec<String>, Option<usize>) {
    let compatible_trucks = instance
        .truck_types
        .iter()
        .filter(|&(_, t_info)| {
            t_info
                .compatible_containers
                .contains(&instance.request_container[pickup])
        })
        .map(|(t, _)| t.to_string())
        .collect::<Vec<String>>();
    assert_ne!(compatible_trucks.len(), 0);
    if compatible_trucks.len() == 1 {
        (compatible_trucks, None)
    } else {
        let indices = sample_weighted(
            rng,
            compatible_trucks.len(),
            |i| (compatible_trucks.len() - i) as f32,
            1,
        )
        .unwrap();
        let index = indices.into_vec()[0];
        (compatible_trucks, Some(index))
    }
}

fn calculate_cost_of_new_route(
    pickup: u16,
    solution: &Solution,
    comp_tr: &[String],
    ind: Option<usize>,
    instance: &InstanceData,
) -> (f32, String) {
    let mut compatible_trucks = comp_tr.to_vec();
    let start_depot = instance.depot_ids[0];
    let end_depot = instance.depot_ids[instance.depot_ids.len() - 1];
    let travel_time = distance(start_depot, pickup, instance)
        + distance(pickup, instance.delivery_of_pickup[&pickup], instance)
        + distance(instance.delivery_of_pickup[&pickup], end_depot, instance);
    let cost = |f: &str| -> f32 {
        instance.truck_types[f].fixed_cost + instance.truck_types[f].routing_cost * travel_time
    };

    // assert_ne!(compatible_trucks.len(), 0);
    if compatible_trucks.len() == 1 {
        return (
            cost(&compatible_trucks[0]),
            compatible_trucks[0].to_string(),
        );
    }

    let index = ind.unwrap();
    compatible_trucks.select_nth_unstable_by_key(index, |f| {
        Reverse((
            get_compatibility(f, &solution.unassigned_requests, instance),
            get_capacity(f, &solution.unassigned_requests, instance),
            // prevalence(f),
            OrderedFloat(-1.0 * cost(f)),
        ))
    });

    (
        cost(&compatible_trucks[index]),
        compatible_trucks[index].to_string(),
    )
}

fn get_cost(ic: &Ins) -> f32 {
    match ic {
        Ins::Insertion(ic) => ic.insertion_cost,
        Ins::NewRoute(ic) => ic.new_route_cost,
    }
}

fn rr_function(i: u16, rr: &HashMap<u16, Vec<Ins>>, k: usize) -> f32 {
    if k > &rr[&i].len() - 1 {
        println!("k: {}, insertion list: {:?}", k, &rr[&i])
    }
    (0..=k)
        .map(|j| get_cost(&rr[&i][j]) - get_cost(&rr[&i][0]))
        .sum()
}

fn get_compatible_routes(solution: &Solution, instance: &InstanceData) -> HashMap<u16, Vec<usize>> {
    let unassigned_container_types = solution
        .unassigned_requests
        .iter()
        .map(|i| &instance.request_container[&i])
        .unique()
        .collect::<Vec<&String>>();

    let container_compatible_routes = unassigned_container_types
        .iter()
        .map(|&ct| {
            (
                ct.to_string(),
                (0..solution.routes.len())
                    .filter(|i| {
                        instance.truck_types[&solution.routes[*i].truck_type]
                            .compatible_containers
                            .contains(ct)
                    })
                    .collect::<Vec<usize>>(),
            )
        })
        .collect::<HashMap<String, Vec<usize>>>();

    let compatible_routes = solution
        .unassigned_requests
        .iter()
        .map(|u| {
            (
                *u,
                container_compatible_routes[&instance.request_container[u]].to_owned(),
            )
        })
        .collect::<HashMap<u16, Vec<usize>>>();
    compatible_routes
}

pub fn regret_insertion<T: Rng>(
    k: usize,
    solution: &mut Solution,
    blink_rate: f32,
    rng: &mut T,
    instance: &InstanceData,
) {
    let c_t: HashMap<u16, (Vec<String>, Option<usize>)> = solution
        .unassigned_requests
        .iter()
        .map(|u| {
            (
                *u,
                identify_compatible_trucks_and_random_select(u, rng, instance),
            )
        })
        .collect();

    let get_results = |u: u16,
                       blink_rate: f32,
                       compatible_routes: &HashMap<u16, Vec<usize>>,
                       sol: &Solution|
     -> Vec<Ins> {
        let compatible_routes = &compatible_routes[&u];
        let mut results_u = (*compatible_routes)
            .par_iter()
            .map(|r| {
                match calculate_best_insertion_with_blinks(
                    u,
                    &sol.routes[*r].stops,
                    &sol.routes[*r].truck_type,
                    sol.routes[*r].cost.routing_cost,
                    blink_rate,
                    instance,
                ) {
                    Some(valid_result) => Ins::Insertion(InsertionData {
                        route_index: Some(*r),
                        insertion_cost: valid_result.0,
                        pickup_index: Some(valid_result.1),
                        delivery_index: Some(valid_result.2),
                    }),
                    None => Ins::Insertion(InsertionData {
                        route_index: None,
                        pickup_index: None,
                        delivery_index: None,
                        insertion_cost: f32::INFINITY,
                    }),
                }
            })
            .collect::<Vec<Ins>>();
        let (nr_cost, nr_truck) =
            calculate_cost_of_new_route(u, &sol, &c_t[&u].0, c_t[&u].1, instance);
        results_u.push(Ins::NewRoute(NewRouteData {
            new_route_cost: nr_cost,
            truck_type: nr_truck,
        }));
        results_u.par_sort_by_cached_key(|r| OrderedFloat(get_cost(r)));
        results_u
    };

    while !solution.unassigned_requests.is_empty() {
        let compatible_routes = get_compatible_routes(solution, instance);
        let results: HashMap<u16, Vec<Ins>> = solution
            .unassigned_requests
            .par_iter()
            .map(|u| {
                (
                    *u,
                    get_results(*u, blink_rate, &compatible_routes, solution),
                )
            })
            .collect();

        let no_of_valid_insertions = |results: &[Ins]| -> usize {
            results
                .iter()
                .filter(|&v| get_cost(v) < f32::INFINITY)
                .count()
        };

        let u_without_enough_insertions = results
            .iter()
            .filter(|&(_, r)| no_of_valid_insertions(r) < k + 1)
            .map(|(u, _)| *u)
            .collect::<Vec<u16>>();

        let selected_u: u16;
        if !u_without_enough_insertions.is_empty() {
            // the selected request is that with the fewest valid insertions; ties are broken
            // by selecting the request with the least insertion cost.
            selected_u = u_without_enough_insertions
                .iter()
                .min_by_key(|&u| {
                    (
                        no_of_valid_insertions(&results[u]),
                        OrderedFloat(get_cost(&results[u][0])),
                    )
                })
                .map(|u| *u)
                .unwrap();
        } else {
            let mut us: Vec<u16> = results.keys().cloned().collect();
            us.select_nth_unstable_by_key(0, |i| {
                Reverse((
                    OrderedFloat(rr_function(*i, &results, k)),
                    OrderedFloat(-1.0 * get_cost(&results[&i][0])),
                ))
            });
            selected_u = us[0];
        }
        let request_data = &results[&selected_u][0];
        insert_request(solution, selected_u, request_data, instance);
    }
}

fn insert_request(
    solution: &mut Solution,
    pickup: u16,
    request_data: &Ins,
    instance: &InstanceData,
) {
    match request_data {
        Ins::Insertion(request_data) => solution.reroute_request(
            pickup,
            request_data.route_index.unwrap(),
            request_data.pickup_index.unwrap(),
            request_data.delivery_index.unwrap(),
            instance,
        ),
        Ins::NewRoute(request_data) => solution.reroute_request_in_independent_route(
            pickup,
            &request_data.truck_type,
            instance,
        ),
    };
}
