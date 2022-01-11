#![allow(non_snake_case)]

use osqp::{CscMatrix, Problem, Settings};

fn main() {
    println!("Quadratic programming example...");

    // Define problem data
    let P :Vec<f64> = [4.0, 1.0, 1.0, 2.0].to_vec();
    let q = &[1.0, 1.0];
    let A = &[[1.0, 1.0],
              [1.0, 0.0],
              [0.0, 1.0]];
    let l = &[1.0, 0.0, 0.0];
    let u = &[1.0, 0.7, 0.7];
    
    // Extract the upper triangular elements of `P`
    let P = CscMatrix::from_column_iter_dense(2,2,P.into_iter()).into_upper_tri();
    
    // Disable verbose output
    let settings = Settings::default()
        .verbose(true);
    
    // Create an OSQP problem
    let mut prob = Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem");
    
    // Solve problem
    let result = prob.solve();
    
    // Print the solution
    println!("{:?}", result.x().expect("failed to solve problem"));

}
