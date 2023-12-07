use crate::instance_data::InstanceData;
use crate::solution::Solution;
use crate::utils::{calculate_index, distance, get_capacity, get_compatibility, visit_time};
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::HashMap;

fn get_routing_cost(route: &[u16], chosen_truck: &str, instance: &InstanceData) -> f32 {
    let routing_cost: f32 = (0..route.len() - 1)
        .map(|k| {
            &instance.truck_types[chosen_truck].routing_cost
                * distance(route[k], route[k + 1], instance)
        })
        .sum();
    let route_cost: f32 = &instance.truck_types[chosen_truck].fixed_cost + routing_cost;
    route_cost
}

fn get_compatible_trucks(i: u16, instance: &InstanceData) -> Vec<&str> {
    let compatible_trucks = instance
        .truck_types
        .iter()
        .filter(|&(_, t)| {
            t.compatible_containers
                .contains(&instance.request_container[&i])
        })
        .map(|(t_type, _)| t_type.as_str())
        .collect::<Vec<&str>>();
    compatible_trucks
}

pub fn naive(instance: &InstanceData) -> Solution {
    let mut rng = rand::thread_rng();
    let requests_ids = instance
        .request_container
        .keys()
        .cloned()
        .collect::<Vec<_>>();

    let mut create_route = |i: u16| -> (&str, Vec<u16>, f32) {
        let compatible_trucks = get_compatible_trucks(i, instance);
        let chosen_truck = compatible_trucks.choose(&mut rng).unwrap();

        let mut route: Vec<u16> = vec![0; 4];
        route[1] = i;
        route[2] = instance.delivery_of_pickup[&i];
        route[3] = instance.depot_ids[1];

        let route_cost = get_routing_cost(&route, *chosen_truck, instance);
        (*chosen_truck, route, route_cost)
    };

    let sol_routes: Vec<(&str, Vec<u16>, f32)> =
        requests_ids.iter().map(|i| create_route(*i)).collect();
    Solution::new(sol_routes, instance)
}

#[allow(dead_code)]
pub fn i1(
    instance: &InstanceData,
    farthest_locations_init: bool,
    alpha1: f32,
    alpha2: f32,
    p: u8,
) -> Solution {
    let mut rng = rand::thread_rng();
    let mut unrouted_requests = instance
        .request_container
        .keys()
        .cloned()
        .collect::<Vec<_>>();

    let init_crit = |r: u16| -> f32 {
        let start_depot = instance.depot_ids[0];
        if farthest_locations_init {
            // farthest
            distance(start_depot, r, instance).max(distance(
                start_depot,
                instance.delivery_of_pickup[&r],
                instance,
            ))
        } else {
            // earliest deadline
            -1.0 * instance.nodes[&instance.delivery_of_pickup[&r]].tw_l
        }
    };

    let mut sol_routes: Vec<(&str, Vec<u16>, f32)> = Vec::new();
    while !unrouted_requests.is_empty() {
        let last_index = unrouted_requests.len() - 1;
        unrouted_requests.select_nth_unstable_by_key(last_index, |r| OrderedFloat(init_crit(*r)));
        let init_request = unrouted_requests.pop().unwrap();

        let mut current_route = CurrentRoute::new(
            vec![
                instance.depot_ids[0],
                init_request,
                instance.delivery_of_pickup[&init_request],
                instance.depot_ids[1],
            ],
            instance,
        );
        let mut compatible_trucks = get_compatible_trucks(init_request, instance);
        let chosen_truck = choose_truck(
            &current_route.stops,
            &unrouted_requests,
            &mut compatible_trucks,
            p,
            &mut rng,
            instance,
        );

        let mut compatible_unrouted_requests = instance
            .request_container
            .iter()
            .filter(|&(r, cont)| {
                unrouted_requests.contains(r)
                    && instance.truck_types[chosen_truck]
                        .compatible_containers
                        .contains(cont)
            })
            .map(|(r, _)| *r)
            .collect::<Vec<_>>();

        loop {
            let best_insertions: HashMap<u16, Option<(CurrentRoute, f32)>> =
                compatible_unrouted_requests
                    .iter()
                    .map(|u| {
                        (
                            *u,
                            current_route.get_best_insertion(*u, alpha1, alpha2, instance),
                        )
                    })
                    .collect();
            if best_insertions
                .values()
                .all(|insertion| insertion.is_none())
            {
                // add route to list
                sol_routes.push((
                    chosen_truck,
                    current_route.stops.clone(),
                    get_routing_cost(&current_route.stops, chosen_truck, instance),
                ));
                break;
            } else {
                let valid_insertions: HashMap<u16, (CurrentRoute, f32)> = best_insertions
                    .iter()
                    .filter(|&(_, insertion)| insertion.is_some())
                    .map(|(u, insertion)| (*u, insertion.clone().unwrap()))
                    .collect();

                let selected_u = valid_insertions
                    .keys()
                    .max_by_key(|&u| {
                        OrderedFloat(
                            distance(instance.depot_ids[0], *u, instance)
                                + distance(*u, instance.delivery_of_pickup[u], instance)
                                + distance(
                                    instance.delivery_of_pickup[u],
                                    instance.depot_ids[1],
                                    instance,
                                )
                                - valid_insertions[u].1,
                        )
                    })
                    .unwrap();

                current_route = valid_insertions[selected_u].0.clone();
                unrouted_requests.retain(|r| r != selected_u);
                compatible_unrouted_requests.retain(|r| r != selected_u);
            }
        }
    }

    Solution::new(sol_routes, instance)
}

fn choose_truck<'a, T: Rng>(
    route: &[u16],
    unrouted_requests: &[u16],
    compatible_trucks: &mut [&'a str],
    p: u8,
    rng: &mut T,
    instance: &InstanceData,
) -> &'a str {
    let index = calculate_index(p, compatible_trucks.len(), rng);
    compatible_trucks.select_nth_unstable_by_key(index, |f| {
        Reverse((
            get_compatibility(f, &unrouted_requests, instance),
            get_capacity(f, &unrouted_requests, instance),
            OrderedFloat(-1.0 * get_routing_cost(route, f, instance)),
        ))
    });
    compatible_trucks[index]
}

#[derive(Debug, Clone)]
struct CurrentRoute {
    stops: Vec<u16>,
    visit_times: Vec<f32>,
}

impl CurrentRoute {
    fn new(stops: Vec<u16>, instance: &InstanceData) -> Self {
        let visit_times = CurrentRoute::get_visit_times(&stops, instance);
        Self { stops, visit_times }
    }

    fn get_visit_times(stops: &[u16], instance: &InstanceData) -> Vec<f32> {
        let mut visit_times: Vec<f32> = vec![0.0; stops.len()];
        visit_times[0] = instance.nodes[&stops[0]].tw_e;
        for i in 1..stops.len() {
            visit_times[i] = visit_time(stops[i], stops[i - 1], visit_times[i - 1], instance);
        }
        visit_times
    }

    fn insert_request(&self, is: usize, r: u16, instance: &InstanceData) -> Option<Self> {
        let mut new_stops = self.stops.clone();
        new_stops.insert(is, instance.delivery_of_pickup[&r]);
        new_stops.insert(is, r);
        let new_visit_times = CurrentRoute::get_visit_times(&new_stops, instance);
        if new_visit_times
            .iter()
            .enumerate()
            .all(|(i, vt)| *vt < instance.nodes[&new_stops[i]].tw_l)
        {
            Some(Self {
                stops: new_stops,
                visit_times: new_visit_times,
            })
        } else {
            None
        }
    }

    fn get_best_insertion(
        &self,
        r: u16,
        alpha1: f32,
        alpha2: f32,
        instance: &InstanceData,
    ) -> Option<(CurrentRoute, f32)> {
        let insertion_locations = self
            .stops
            .iter()
            .enumerate()
            .filter(|&(_, r)| *r == instance.depot_ids[0] || instance.delivery_ids.contains(r))
            .map(|(i, _)| i + 1)
            .collect::<Vec<usize>>();

        let mut valid_routes: Vec<(CurrentRoute, f32)> = Vec::new();
        for index in insertion_locations {
            let opt_new_route = self.insert_request(index, r, instance);
            if opt_new_route.is_some() {
                let new_route = opt_new_route.unwrap();
                let c11 = distance(new_route.stops[index - 1], new_route.stops[index], instance)
                    + distance(new_route.stops[index], new_route.stops[index + 1], instance)
                    + distance(
                        new_route.stops[index + 1],
                        new_route.stops[index + 2],
                        instance,
                    )
                    - distance(
                        new_route.stops[index - 1],
                        new_route.stops[index + 2],
                        instance,
                    );

                let c12 = new_route.visit_times[index + 2] - self.visit_times[index];
                let c1 = alpha1 * c11 + alpha2 * c12;
                valid_routes.push((new_route, c1));
            }
        }

        if !valid_routes.is_empty() {
            let best_valid_route = valid_routes
                .iter()
                .min_by_key(|&route| OrderedFloat(route.1))
                .unwrap();
            Some(best_valid_route.clone())
        } else {
            None
        }
    }
}
