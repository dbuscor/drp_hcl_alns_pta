extern crate core;

mod alns;
mod alns_ops;
mod initial_solution_creator;
mod instance_data;
mod solution;
mod utils;

use crate::alns::ALNS;
use crate::initial_solution_creator::naive;
use instance_data::read_instance;
use std::time::Instant;

fn main() {
    let instance_file_path = "/path/to/instance.json";
    let instance_data = read_instance(instance_file_path).expect("Error reading file");
    let solution = naive(&instance_data);
    println!("{:.2?}", solution.objective());
    let mut alns = ALNS::builder(
        &instance_data,
        60.0,
        0.0108,
        360,
        30,
        [9.3280, 5.0517, 1.1480],
        0.7955,
        true,
    )
    .set_random_removal(0.1152)
    .set_shaw_removal(0.1030, [7.8448, 4.9373, 1.3596])
    .set_sisrl_removal(0.1436)
    .set_regret1(0.9771)
    .build();
    let now = Instant::now();
    let (updated_solution, _history) = alns.run(&solution);
    let elapsed = now.elapsed();
    println!("{:.2?}", updated_solution.objective());
    println!("{:?}", updated_solution.get_fleet_composition());
    println!("Elapsed: {:.2?}", elapsed);
}
