use crate::instance_data::InstanceData;
use crate::utils::distance;
use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct RequestPosition {
    pub route_id: usize,
    pub pickup_index: usize,
    pub delivery_index: usize,
}

#[derive(Clone, Debug)]
pub struct Cost {
    pub fixed_cost: f32,
    pub routing_cost: f32,
}

impl Cost {
    pub fn total(&self) -> f32 {
        self.fixed_cost + self.routing_cost
    }
}

#[derive(Clone, Debug)]
pub struct Route {
    pub truck_type: String,
    pub cost: Cost,
    pub stops: Vec<u16>,
    pub stops_states: Vec<u16>,
    pub stops_visit_times: Vec<f32>,
}

impl Route {
    pub fn new(
        truck_type: String,
        cost: Cost,
        stops: Vec<u16>,
        stops_states: Vec<u16>,
        stops_visit_times: Vec<f32>,
    ) -> Self {
        assert_eq!(stops.len(), stops_states.len());
        assert_eq!(stops.len(), stops_visit_times.len());
        Self {
            truck_type,
            cost,
            stops,
            stops_states,
            stops_visit_times,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub routes: Vec<Route>,
    pub request_position: HashMap<u16, RequestPosition>,
    pub cost: Cost,
    pub unassigned_requests: Vec<u16>,
}

impl Solution {
    pub fn new(sol_routes: Vec<(&str, Vec<u16>, f32)>, instance: &InstanceData) -> Self {
        let mut routes: Vec<Route> = Vec::new();

        for r in &sol_routes {
            let t_type: &str = &(*r).0;
            let cost: Cost = Cost {
                fixed_cost: instance.truck_types[t_type].fixed_cost,
                routing_cost: &(*r).2 - instance.truck_types[t_type].fixed_cost,
            };
            let stops = ((*r).1).clone();
            let (stops_states, stops_visit_times) = Solution::set_stops(t_type, &(*r).1, &instance);
            let route: Route = Route::new(
                t_type.to_string(),
                cost,
                stops,
                stops_states,
                stops_visit_times,
            );
            routes.push(route);
        }

        let request_position: HashMap<u16, RequestPosition> =
            Solution::set_requests_positions(&routes, &instance);

        let cost: Cost = Cost {
            fixed_cost: routes.iter().map(|r| r.cost.fixed_cost).sum(),
            routing_cost: routes.iter().map(|r| r.cost.routing_cost).sum(),
        };

        let unassigned_requests: Vec<u16> = Vec::new();
        Self {
            routes,
            request_position,
            cost,
            unassigned_requests,
        }
    }

    fn set_stops(truck_type: &str, r: &[u16], instance: &InstanceData) -> (Vec<u16>, Vec<f32>) {
        let start_transition: Vec<&(u16, u16, String)> = instance.truck_types[truck_type]
            .transitions
            .iter()
            .filter(|&t| t.2 == "start")
            .collect();
        assert_eq!(start_transition.len(), 1);
        let start_state = (*start_transition[0]).1;

        //let mut stops: Vec<u16> = vec![0];
        let mut stops_states: Vec<u16> = vec![start_state];
        let mut stops_visit_times: Vec<f32> = vec![0.0];

        for i in 1..r.len() {
            let stop = &r[i];
            stops_states.push(Solution::new_state(
                &stops_states[i - 1],
                &instance.nodes[stop].r,
                truck_type,
                &instance,
            ));
            stops_visit_times.push(Solution::visit_time(
                stop,
                &r[i - 1],
                &stops_visit_times[i - 1],
                &instance,
            ));
        }

        (stops_states, stops_visit_times)
    }

    fn new_state(
        previous_state: &u16,
        customer_request: &str,
        truck_type: &str,
        instance: &InstanceData,
    ) -> u16 {
        let transition: Vec<&(u16, u16, String)> = instance.truck_types[truck_type]
            .transitions
            .iter()
            .filter(|&t| t.0 == *previous_state && t.2 == customer_request)
            .collect();
        assert_eq!(transition.len(), 1);
        (*transition[0]).1
    }

    fn visit_time(
        i: &u16,
        i_minus_1: &u16,
        time_at_i_minus_1: &f32,
        instance: &InstanceData,
    ) -> f32 {
        if *i == 0 {
            0.0
        } else {
            instance.nodes[i].tw_e.max(
                *time_at_i_minus_1
                    + instance.nodes[i_minus_1].st
                    + distance(*i_minus_1, *i, &instance),
            )
        }
    }

    fn set_requests_positions(
        routes: &Vec<Route>,
        instance: &InstanceData,
    ) -> HashMap<u16, RequestPosition> {
        let mut requests_positions: HashMap<u16, RequestPosition> =
            HashMap::with_capacity(instance.pickup_ids.len());

        for &p in instance.pickup_ids.iter() {
            requests_positions.insert(
                p,
                RequestPosition {
                    route_id: usize::MAX,
                    pickup_index: usize::MAX,
                    delivery_index: usize::MAX,
                },
            );
        }

        for (i, route) in routes.iter().enumerate() {
            for (j, stop) in route.stops.iter().enumerate() {
                if instance.pickup_ids.contains(stop) {
                    let r_p = requests_positions.get_mut(stop).unwrap();
                    r_p.route_id = i;
                    r_p.pickup_index = j;
                } else if instance.delivery_ids.contains(stop) {
                    let pickup = &instance.pickup_of_delivery[stop];
                    let r_p = requests_positions.get_mut(pickup).unwrap();
                    r_p.delivery_index = j;
                }
            }
        }

        requests_positions
    }

    fn update_route_data(&mut self, route_id: usize, instance: &InstanceData) {
        // The stops should have already been removed/added to the route before calling this function
        let truck_type = &self.routes[route_id].truck_type;
        let stops = &self.routes[route_id].stops;

        let (stops_states, stops_visit_times) = Solution::set_stops(truck_type, stops, instance);

        self.routes[route_id].cost.routing_cost =
            Solution::set_routing_cost(truck_type, stops, instance);

        self.routes[route_id].stops_states = stops_states;
        self.routes[route_id].stops_visit_times = stops_visit_times;
    }

    pub fn remove_requests(&mut self, request_list: &[u16], instance: &InstanceData) {
        for r in request_list {
            let request_positions = &self.request_position[r];
            let route_id = request_positions.route_id;
            let pickup_index = request_positions.pickup_index;
            let delivery_index = request_positions.delivery_index;

            let mutable_route = &mut self.routes[route_id];
            let sequence_of_stops = &mut mutable_route.stops;
            let sequence_of_visit_times = &mut mutable_route.stops_visit_times;
            let sequence_of_states = &mut mutable_route.stops_states;

            let short_route = vec![
                instance.depot_ids[0],
                *r,
                instance.delivery_of_pickup[r],
                instance.depot_ids[instance.depot_ids.len() - 1],
            ];

            if !sequence_of_stops.iter().eq(short_route.iter()) {
                let stops_between_pickup_and_delivery = sequence_of_stops
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| pickup_index < i && i < delivery_index)
                    .map(|(_, e)| *e)
                    .collect::<Vec<u16>>();

                for s in &stops_between_pickup_and_delivery {
                    if instance.pickup_ids.contains(s) {
                        let pos = self.request_position.get_mut(s).unwrap();
                        pos.pickup_index -= 1;
                    } else {
                        let s_pickup = instance.pickup_of_delivery[s];
                        let pos = self.request_position.get_mut(&s_pickup).unwrap();
                        pos.delivery_index -= 1;
                    }
                }

                let stops_after_delivery = sequence_of_stops
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| delivery_index < i && i < sequence_of_stops.len() - 1)
                    .map(|(_, e)| *e)
                    .collect::<Vec<u16>>();

                for s in &stops_after_delivery {
                    if instance.pickup_ids.contains(s) {
                        let pos = self.request_position.get_mut(s).unwrap();
                        pos.pickup_index -= 2;
                    } else {
                        let s_pickup = instance.pickup_of_delivery[s];
                        let pos = self.request_position.get_mut(&s_pickup).unwrap();
                        pos.delivery_index -= 2;
                    }
                }
            }
            sequence_of_stops.remove(delivery_index);
            sequence_of_stops.remove(pickup_index);

            sequence_of_visit_times.remove(delivery_index);
            sequence_of_visit_times.remove(pickup_index);

            sequence_of_states.remove(delivery_index);
            sequence_of_states.remove(pickup_index);

            if sequence_of_stops[1] != instance.depot_ids[instance.depot_ids.len() - 1] {
                self.update_route_data(route_id, instance);
            }

            self.request_position.remove(r);
            self.unassigned_requests.push(*r);
        }

        let mut update_r_indices = false;
        let mut empty_routes_indices = self
            .routes
            .iter()
            .enumerate()
            .filter(|&(_, r)| (*r).stops[1] == instance.depot_ids[instance.depot_ids.len() - 1])
            .map(|(i, _)| i)
            .collect::<Vec<usize>>();

        if empty_routes_indices.len() > 0 {
            update_r_indices = true;
            empty_routes_indices.sort_unstable_by(|a, b| b.cmp(a));
            for r in empty_routes_indices {
                self.routes.remove(r);
            }
        }

        if update_r_indices {
            self.update_route_indices(instance); // update only if there were routes removed
        }

        self.update_total_cost();
    }

    pub fn start_new_route(truck_type: &str, stops: &[u16], instance: &InstanceData) -> Route {
        let (stops_states, stops_visit_times) = Solution::set_stops(truck_type, stops, instance);
        let cost = Cost {
            fixed_cost: instance.truck_types[truck_type].fixed_cost,
            routing_cost: Solution::set_routing_cost(truck_type, stops, instance),
        };
        let route = Route::new(
            truck_type.to_string(),
            cost,
            stops.to_vec(),
            stops_states,
            stops_visit_times,
        );
        route
    }

    pub fn reroute_request_in_independent_route(
        &mut self,
        pickup: u16,
        truck_type: &str,
        instance: &InstanceData,
    ) {
        assert!(instance.pickup_ids.contains(&pickup));
        assert!(self.unassigned_requests.contains(&pickup));
        assert!(instance.truck_types[truck_type]
            .compatible_containers
            .contains(&instance.request_container[&pickup]));
        let stops = vec![
            instance.depot_ids[0],
            pickup,
            instance.delivery_of_pickup[&pickup],
            instance.depot_ids[instance.depot_ids.len() - 1],
        ];
        let new_route = Solution::start_new_route(truck_type, &stops, instance);
        self.cost.fixed_cost += new_route.cost.fixed_cost;
        self.cost.routing_cost += new_route.cost.routing_cost;
        self.routes.push(new_route);
        self.request_position.insert(
            pickup,
            RequestPosition {
                route_id: self.routes.len() - 1,
                pickup_index: 1,
                delivery_index: 2,
            },
        );
        self.unassigned_requests.retain(|&x| x != pickup);
    }

    pub fn reroute_request(
        &mut self,
        pickup: u16,
        route_index: usize,
        pickup_index: usize,
        delivery_index: usize,
        instance: &InstanceData,
    ) {
        assert!(instance.pickup_ids.contains(&pickup));
        assert!(self.unassigned_requests.contains(&pickup));
        assert!(instance.truck_types[&self.routes[route_index].truck_type]
            .compatible_containers
            .contains(&instance.request_container[&pickup]));

        self.routes[route_index]
            .stops
            .insert(delivery_index, instance.delivery_of_pickup[&pickup]);
        self.routes[route_index].stops.insert(pickup_index, pickup);
        self.update_route_data(route_index, instance);
        self.request_position.insert(
            pickup,
            RequestPosition {
                route_id: route_index,
                pickup_index,
                delivery_index: delivery_index + 1,
            },
        );

        let stops_between_pickup_and_delivery = self.routes[route_index]
            .stops
            .iter()
            .enumerate()
            .filter(|&(i, _)| pickup_index < i && i < delivery_index + 1)
            .map(|(_, e)| *e)
            .collect::<Vec<u16>>();

        for s in &stops_between_pickup_and_delivery {
            if instance.pickup_ids.contains(s) {
                let pos = self.request_position.get_mut(s).unwrap();
                pos.pickup_index += 1;
            } else {
                let s_pickup = instance.pickup_of_delivery[s];
                let pos = self.request_position.get_mut(&s_pickup).unwrap();
                pos.delivery_index += 1;
            }
        }

        let stops_after_delivery = self.routes[route_index]
            .stops
            .iter()
            .enumerate()
            .filter(|&(i, _)| {
                delivery_index + 1 < i && i < self.routes[route_index].stops.len() - 1
            })
            .map(|(_, e)| *e)
            .collect::<Vec<u16>>();

        for s in &stops_after_delivery {
            if instance.pickup_ids.contains(s) {
                let pos = self.request_position.get_mut(s).unwrap();
                pos.pickup_index += 2;
            } else {
                let s_pickup = instance.pickup_of_delivery[s];
                let pos = self.request_position.get_mut(&s_pickup).unwrap();
                pos.delivery_index += 2;
            }
        }
        self.update_total_cost();
        self.unassigned_requests.retain(|&x| x != pickup);
    }

    pub fn get_visit_times(&self, request: u16) -> (f32, f32) {
        let request_position = &self.request_position[&request];
        let route_id = request_position.route_id;
        let pickup_index = request_position.pickup_index;
        let delivery_index = request_position.delivery_index;
        let vt_pickup = self.routes[route_id].stops_visit_times[pickup_index];
        let vt_delivery = self.routes[route_id].stops_visit_times[delivery_index];
        (vt_pickup, vt_delivery)
    }

    fn update_route_indices(&mut self, instance: &InstanceData) {
        for route_index in 0..self.routes.len() {
            let route = &self.routes[route_index];
            for stop_index in 1..route.stops.len() - 1 {
                let stop = route.stops[stop_index];
                if instance.pickup_ids.contains(&stop) {
                    let pos = self.request_position.get_mut(&stop).unwrap();
                    pos.route_id = route_index;
                }
            }
        }
    }

    pub fn update_total_cost(&mut self) {
        self.cost.fixed_cost = self.routes.iter().map(|r| r.cost.fixed_cost).sum();
        self.cost.routing_cost = self.routes.iter().map(|r| r.cost.routing_cost).sum();
    }

    pub fn set_routing_cost(truck_type: &str, stops: &[u16], instance: &InstanceData) -> f32 {
        let mut routing_cost: f32 = 0.0;
        let truck_rc = instance.truck_types[truck_type].routing_cost;
        for i in 0..stops.len() - 1 {
            routing_cost += truck_rc * distance(stops[i], stops[i + 1], instance);
        }
        routing_cost
    }

    pub fn objective(&self) -> f32 {
        self.cost.total()
    }

    #[allow(dead_code)]
    pub fn convert_to_lists(&self) -> Vec<(String, Vec<u16>, f32)> {
        self.routes
            .iter()
            .map(|r| (r.truck_type.to_string(), r.stops.clone(), r.cost.total()))
            .collect::<Vec<(String, Vec<u16>, f32)>>()
    }

    pub fn get_fleet_composition(&self) -> HashMap<String, usize> {
        if !self.routes.iter().all(|r| r.stops.len() >= 4) {
            panic!("there are routes with fewer than four stops!")
        }
        let routes_truck_types = self
            .routes
            .iter()
            .map(|r| r.truck_type.clone())
            .collect::<Vec<String>>();

        let mut truck_types_count: HashMap<String, usize> = HashMap::new();
        for tt in routes_truck_types {
            *truck_types_count.entry(tt).or_default() += 1;
        }
        truck_types_count
    }
}

impl PartialEq<Self> for Solution {
    fn eq(&self, other: &Self) -> bool {
        OrderedFloat(self.cost.total()) == OrderedFloat(other.cost.total())
    }
}
impl Eq for Solution {}

impl Ord for Solution {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.cost.total()).cmp(&OrderedFloat(other.cost.total()))
    }
}

impl PartialOrd for Solution {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
