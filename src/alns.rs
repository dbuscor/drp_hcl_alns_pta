use crate::alns_ops::{random_removal, regret_insertion, shaw_removal, sisrs_removal};
use crate::instance_data::InstanceData;
use crate::solution::Solution;
use cached::Cached;
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;
use std::collections::{BTreeSet, HashSet};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Destruction {
    Shaw,
    Random,
    Sisrl,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Repair {
    Regret0,
    Regret1,
    Regret2,
    Regret3,
}

#[derive(Debug, Clone, Copy)]
struct OptData {
    weight: f32,
    score: f32,
    usages: usize,
}

struct WeightScheme {
    score_parameters: [f32; 3],
    d_data: Vec<OptData>,
    r_data: Vec<OptData>,
    forgetting_factor: f32,
}

impl WeightScheme {
    fn new(
        score_parameters: [f32; 3],
        num_destroy: usize,
        num_repair: usize,
        forgetting_factor: f32,
    ) -> Self {
        let d_data: Vec<OptData> = vec![
            OptData {
                weight: 1.0,
                score: 0.0,
                usages: 0
            };
            num_destroy
        ];
        let r_data: Vec<OptData> = vec![
            OptData {
                weight: 1.0,
                score: 0.0,
                usages: 0
            };
            num_repair
        ];

        WeightScheme {
            score_parameters,
            d_data,
            r_data,
            forgetting_factor,
        }
    }

    fn select_operator<T: Rng>(&self, rng: &mut T) -> (usize, usize) {
        let mut selection = |data: &[OptData]| -> usize {
            let sum: f32 = data.iter().map(|d| d.weight).sum();
            let weights: Vec<f32> = data.iter().map(|d| d.weight / sum).collect();
            let dist = WeightedIndex::new(&weights).unwrap();
            let indices: Vec<usize> = (0..weights.len()).collect();
            indices[dist.sample(rng)]
        };
        let d_index = selection(&self.d_data);
        let r_index = selection(&self.r_data);
        (d_index, r_index)
    }

    fn update_scores(&mut self, d_index: usize, r_index: usize, s_index: usize) {
        self.d_data[d_index].score += self.score_parameters[s_index];
        self.r_data[r_index].score += self.score_parameters[s_index];
        self.d_data[d_index].usages += 1;
        self.r_data[r_index].usages += 1;
    }

    fn update_weights(&mut self) {
        // A block has ended; weights are updated and scores and number of usages are reset.
        for opt_data in &mut self.d_data {
            let frac = if opt_data.score > 0.0 {
                opt_data.score / opt_data.usages as f32
            } else {
                0.0
            };
            opt_data.weight =
                opt_data.weight * (1.0 - self.forgetting_factor) + self.forgetting_factor * frac;
            opt_data.score = 0.0;
            opt_data.usages = 0;
        }
        for opt_data in &mut self.r_data {
            let frac = if opt_data.score > 0.0 {
                opt_data.score / opt_data.usages as f32
            } else {
                0.0
            };
            opt_data.weight =
                opt_data.weight * (1.0 - self.forgetting_factor) + self.forgetting_factor * frac;
            opt_data.score = 0.0;
            opt_data.usages = 0;
        }
    }
}

struct HistoryOfSolutions {
    current_best: f32,
    history_of_best: Vec<f32>,
    no_of_records: usize,
}

impl HistoryOfSolutions {
    fn new(initial_best: f32, capacity: usize) -> Self {
        let mut history_of_best: Vec<f32> = vec![-1.0; capacity];
        history_of_best[0] = initial_best;
        Self {
            current_best: initial_best,
            history_of_best,
            no_of_records: 1,
        }
    }

    fn push_best(&mut self) {
        if self.no_of_records < self.history_of_best.len() {
            self.history_of_best[self.no_of_records] = self.current_best;
            self.no_of_records += 1;
            // println!(
            //     "saved record {}, history: {:?}",
            //     self.current_best, self.history_of_best
            // );
        }
    }

    fn set_last_best(&mut self, last_best_cost: f32) {
        let last_index = self.history_of_best.len() - 1;
        self.history_of_best[last_index] = last_best_cost;
        // println!(
        //     "last saved record {}, history: {:?}",
        //     last_best_cost, self.history_of_best
        // );
    }
}

pub struct AlnsBuilder<'a> {
    instance: &'a InstanceData,
    max_runtime: f32,
    acceptance_threshold: f32,
    max_cnis: u64,
    cs_to_update_weights: u64,
    scores: [f32; 3],
    forgetting_factor: f32,
    verbose: bool,
    destruction_ops: Vec<Destruction>,
    random_q: Option<f32>,
    shaw_q: Option<f32>,
    shaw_weights: Option<[f32; 3]>,
    sisrl_q: Option<f32>,
    repair_ops: Vec<Repair>,
    regret0_blink_rate: Option<f32>,
    regret1_blink_rate: Option<f32>,
    regret2_blink_rate: Option<f32>,
    regret3_blink_rate: Option<f32>,
}

impl<'a> AlnsBuilder<'a> {
    pub fn new(
        instance: &'a InstanceData,
        max_runtime: f32,
        acceptance_threshold: f32,
        max_cnis: u64,
        cs_to_update_weights: u64,
        scores: [f32; 3],
        forgetting_factor: f32,
        verbose: bool,
    ) -> Self {
        Self {
            instance,
            max_runtime,
            acceptance_threshold,
            max_cnis,
            cs_to_update_weights,
            scores,
            forgetting_factor,
            verbose,
            destruction_ops: vec![],
            random_q: None,
            shaw_q: None,
            shaw_weights: None,
            sisrl_q: None,
            repair_ops: vec![],
            regret0_blink_rate: None,
            regret1_blink_rate: None,
            regret2_blink_rate: None,
            regret3_blink_rate: None,
        }
    }

    pub fn set_random_removal(mut self, q: f32) -> Self {
        self.destruction_ops.push(Destruction::Random);
        self.random_q = Some(q);
        self
    }

    pub fn set_shaw_removal(mut self, q: f32, weights: [f32; 3]) -> Self {
        self.destruction_ops.push(Destruction::Shaw);
        self.shaw_q = Some(q);
        self.shaw_weights = Some(weights);
        self
    }

    pub fn set_sisrl_removal(mut self, q: f32) -> Self {
        self.destruction_ops.push(Destruction::Sisrl);
        self.sisrl_q = Some(q);
        self
    }

    pub fn set_regret0(mut self, blink_rate: f32) -> Self {
        self.repair_ops.push(Repair::Regret0);
        self.regret0_blink_rate = Some(blink_rate);
        self
    }

    pub fn set_regret1(mut self, blink_rate: f32) -> Self {
        self.repair_ops.push(Repair::Regret1);
        self.regret1_blink_rate = Some(blink_rate);
        self
    }

    pub fn set_regret2(mut self, blink_rate: f32) -> Self {
        self.repair_ops.push(Repair::Regret2);
        self.regret2_blink_rate = Some(blink_rate);
        self
    }

    pub fn set_regret3(mut self, blink_rate: f32) -> Self {
        self.repair_ops.push(Repair::Regret3);
        self.regret3_blink_rate = Some(blink_rate);
        self
    }

    pub fn build(self) -> ALNS<'a> {
        if self.destruction_ops.is_empty() || self.repair_ops.is_empty() {
            panic!("At least one destruction op and one repair op must be provided!");
        }

        let des_set = self.destruction_ops.iter().cloned().collect::<HashSet<_>>();
        let rep_set = self.repair_ops.iter().cloned().collect::<HashSet<_>>();
        if self.destruction_ops.len() != des_set.len() || self.repair_ops.len() != rep_set.len() {
            panic!("Each op can be selected only once!");
        }

        let weighting_scheme = WeightScheme::new(
            self.scores,
            self.destruction_ops.len(),
            self.repair_ops.len(),
            self.forgetting_factor,
        );
        ALNS {
            instance: self.instance,
            max_runtime: self.max_runtime,
            cs_to_update_weights: self.cs_to_update_weights,
            acceptance_threshold: self.acceptance_threshold,
            max_cnis: self.max_cnis,
            verbose: self.verbose,
            weighting_scheme,
            destruction_ops: self.destruction_ops,
            random_q: self.random_q,
            shaw_q: self.shaw_q,
            shaw_weights: self.shaw_weights,
            sisrl_q: self.sisrl_q,
            repair_ops: self.repair_ops,
            regret0_blink_rate: self.regret0_blink_rate,
            regret1_blink_rate: self.regret1_blink_rate,
            regret2_blink_rate: self.regret2_blink_rate,
            regret3_blink_rate: self.regret3_blink_rate,
        }
    }
}

pub struct ALNS<'a> {
    instance: &'a InstanceData,
    max_runtime: f32,
    acceptance_threshold: f32,
    max_cnis: u64,
    cs_to_update_weights: u64,
    weighting_scheme: WeightScheme,
    verbose: bool,
    destruction_ops: Vec<Destruction>,
    random_q: Option<f32>,
    shaw_q: Option<f32>,
    shaw_weights: Option<[f32; 3]>,
    sisrl_q: Option<f32>,
    repair_ops: Vec<Repair>,
    regret0_blink_rate: Option<f32>,
    regret1_blink_rate: Option<f32>,
    regret2_blink_rate: Option<f32>,
    regret3_blink_rate: Option<f32>,
}

impl<'a> ALNS<'a> {
    pub fn builder(
        instance: &'a InstanceData,
        max_runtime: f32,
        acceptance_threshold: f32,
        max_cnis: u64,
        cs_to_update_weights: u64,
        scores: [f32; 3],
        forgetting_factor: f32,
        verbose: bool,
    ) -> AlnsBuilder {
        AlnsBuilder::new(
            instance,
            max_runtime,
            acceptance_threshold,
            max_cnis,
            cs_to_update_weights,
            scores,
            forgetting_factor,
            verbose,
        )
    }

    fn destruct<T: Rng>(&self, current: &Solution, op: &Destruction, rng: &mut T) -> Solution {
        match op {
            Destruction::Shaw => shaw_removal(
                current,
                self.shaw_q.unwrap(),
                &self.shaw_weights.unwrap(),
                rng,
                self.instance,
            ),
            Destruction::Random => {
                random_removal(current, self.random_q.unwrap(), rng, self.instance)
            }
            Destruction::Sisrl => sisrs_removal(current, self.sisrl_q.unwrap(), rng, self.instance),
        }
    }

    fn repair<T: Rng>(&self, candidate: &mut Solution, op: &Repair, rng: &mut T) {
        match op {
            Repair::Regret0 => regret_insertion(
                0,
                candidate,
                self.regret0_blink_rate.unwrap(),
                rng,
                self.instance,
            ),
            Repair::Regret1 => regret_insertion(
                1,
                candidate,
                self.regret1_blink_rate.unwrap(),
                rng,
                self.instance,
            ),
            Repair::Regret2 => regret_insertion(
                2,
                candidate,
                self.regret2_blink_rate.unwrap(),
                rng,
                self.instance,
            ),
            Repair::Regret3 => regret_insertion(
                3,
                candidate,
                self.regret3_blink_rate.unwrap(),
                rng,
                self.instance,
            ),
        }
    }

    fn get_stops(solution: &Solution) -> BTreeSet<Vec<u16>> {
        solution
            .routes
            .iter()
            .map(|r| r.stops.clone())
            .collect::<BTreeSet<Vec<u16>>>()
    }

    pub fn run(&mut self, initial_solution: &Solution) -> (Solution, Vec<f32>) {
        let mut rng = rand::thread_rng();

        let history_of_solutions = Arc::new(Mutex::new(HistoryOfSolutions::new(
            initial_solution.cost.total(),
            101,
        )));
        ALNS::schedule_history_updates(&history_of_solutions, self.max_runtime / 100.0);

        let mut current = initial_solution.clone();
        let mut best = initial_solution.clone();

        let mut t = self.acceptance_threshold;
        let runtime = Instant::now();
        let mut consecutive_solutions: u64 = 0;
        let mut consecutive_non_improving_solutions: u64 = 0;

        loop {
            let (d_index, r_index) = self.weighting_scheme.select_operator(&mut rng);
            let d_operator = &self.destruction_ops[d_index];
            let r_operator = &self.repair_ops[r_index];

            if self.verbose {
                println!(
                    "{:.2?}, {:?}, {:.2?}, {:.2?}",
                    runtime.elapsed().as_secs_f32(),
                    t,
                    current.cost.total(),
                    best.cost.total(),
                );
            }

            let mut candidate = self.destruct(&current, d_operator, &mut rng);
            self.repair(&mut candidate, r_operator, &mut rng);
            consecutive_solutions += 1;

            let best_cost = best.cost.total();
            let (new_best, new_current, score_index) = ALNS::eval(best, current, candidate, t);

            best = new_best;
            current = new_current;

            if best.cost.total() < best_cost {
                t = self.acceptance_threshold
                    * (1.0
                        - (self.max_runtime.min(runtime.elapsed().as_secs_f32())
                            / self.max_runtime));
                consecutive_non_improving_solutions = 0;
            } else {
                consecutive_non_improving_solutions += 1;
                if consecutive_non_improving_solutions == self.max_cnis {
                    t = self.acceptance_threshold
                        * (1.0
                            - (self.max_runtime.min(runtime.elapsed().as_secs_f32())
                                / self.max_runtime));
                    consecutive_non_improving_solutions = 0;
                } else {
                    t -= t / (self.max_cnis - consecutive_non_improving_solutions) as f32;
                }
            }

            history_of_solutions.lock().unwrap().current_best = best.cost.total();

            self.weighting_scheme
                .update_scores(d_index, r_index, score_index);
            if consecutive_solutions == self.cs_to_update_weights {
                self.weighting_scheme.update_weights();
                consecutive_solutions = 0;
            }

            if runtime.elapsed().as_secs_f32() >= self.max_runtime {
                break;
            }
        }

        thread::sleep(Duration::from_secs_f32(1.0));
        history_of_solutions
            .lock()
            .unwrap()
            .set_last_best(best.cost.total());
        let final_history = history_of_solutions.lock().unwrap().history_of_best.clone();
        ALNS::reset_cache();
        (best, final_history)
    }

    fn eval(
        best: Solution,
        mut current: Solution,
        candidate: Solution,
        t: f32,
    ) -> (Solution, Solution, usize) {
        /*
        0: index of best score
        1: index of second best score
        2: index of worst score
        */

        let mut score_index: usize = 2;
        if candidate.objective() < best.objective() {
            return (candidate.clone(), candidate.clone(), 0);
        }

        if (candidate.objective() - best.objective()) / candidate.objective() < t {
            score_index = 1;
            current = candidate.clone();
        }

        if (current.objective() - best.objective()) / current.objective() > t {
            current = best.clone();
        }

        (best, current, score_index)
    }

    fn reset_cache() {
        let mut cache1 = crate::alns_ops::ADJACENT_REQUESTS.lock().unwrap();
        let mut cache2 = crate::alns_ops::CALCULATE_BEST_INSERTION.lock().unwrap();
        let mut cache3 = crate::utils::DISTANCE.lock().unwrap();
        let mut cache4 = crate::utils::CHECK_TRANSITIONS.lock().unwrap();
        cache1.cache_reset();
        cache2.cache_reset();
        cache3.cache_reset();
        cache4.cache_reset();
    }

    fn schedule_history_updates(
        history_of_solutions: &Arc<Mutex<HistoryOfSolutions>>,
        duration: f32,
    ) {
        let hs = Arc::clone(history_of_solutions);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_secs_f32(duration));
            hs.lock().unwrap().push_best();
        });
    }

    /* start: RRT run and eval */
    pub fn run_rrt(&mut self, initial_solution: &Solution) -> Solution {
        let mut rng = rand::thread_rng();
        let mut t = self.acceptance_threshold;

        let mut current = initial_solution.clone();
        let mut best = initial_solution.clone();

        let runtime = Instant::now();
        let mut consecutive_solutions: u64 = 0;
        loop {
            let iter_duration = Instant::now();
            let (d_index, r_index) = self.weighting_scheme.select_operator(&mut rng);
            let d_operator = &self.destruction_ops[d_index];
            let r_operator = &self.repair_ops[r_index];

            if self.verbose {
                println!(
                    "{:.2?}, {:?}, {:.2?}, {:.2?}",
                    runtime.elapsed().as_secs_f32(),
                    t,
                    current.cost.total(),
                    best.cost.total(),
                );
            }

            let mut candidate = self.destruct(&current, d_operator, &mut rng);
            self.repair(&mut candidate, r_operator, &mut rng);
            consecutive_solutions += 1;

            let (new_best, new_current, score_index) = ALNS::eval_rrt(best, current, candidate, t);
            best = new_best;
            current = new_current;

            self.weighting_scheme
                .update_scores(d_index, r_index, score_index);
            if consecutive_solutions == self.cs_to_update_weights {
                self.weighting_scheme.update_weights();
                consecutive_solutions = 0;
            }

            t -= self.acceptance_threshold
                * (iter_duration.elapsed().as_secs_f32() / self.max_runtime);

            if runtime.elapsed().as_secs_f32() >= self.max_runtime {
                break;
            }
        }

        ALNS::reset_cache();
        best
    }

    fn eval_rrt(
        best: Solution,
        mut current: Solution,
        candidate: Solution,
        t: f32,
    ) -> (Solution, Solution, usize) {
        let mut score_index: usize = 2;
        if candidate.objective() < best.objective() {
            return (candidate.clone(), candidate.clone(), 0);
        }

        if (candidate.objective() - best.objective()) / candidate.objective() < t {
            score_index = 1;
            current = candidate.clone();
        }

        (best, current, score_index)
    }
}
