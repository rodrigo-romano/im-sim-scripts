//
// -----------------------------------------------------------------
// First version of an AcO standalone implementation
// -----------------------------------------------------------------
// It does not implement the mean slop removal feature,
// neither the rho_3 update iteration.
// April, 2021
//
// Rodrigo A. Romano
//

use serde::Deserialize;
use serde_pickle as pickle;
use std::{fs::File, io::BufReader};
//use rulinalg::matrix::{Matrix, BaseMatrix};
use nalgebra::{SVector,SMatrix,DMatrix};
use osqp::{CscMatrix, Problem, Settings};


// Matrix type definitions
type Matrix_ncxnc = SMatrix<f64, 271, 271>;
type Matrix_nsxnc = SMatrix<f64, 7360, 271>;
type DynMatrix = DMatrix<f64>;
type Vector_ns = SMatrix<f64, 7360, 1>;

// AcO data structure
#[derive(Deserialize)]
struct QPData {
    #[serde(rename = "D")]
    dmat: Vec<f64>,
    #[serde(rename = "W2")]
    w2: Vec<f64>,
    #[serde(rename = "W3")]
    w3: Vec<f64>,
    #[serde(rename = "K")]
    k: f64,
    #[serde(rename = "wfsMask")]
    wfs_mask: Vec<Vec<bool>>,
    umin: Vec<f64>,
    umax: Vec<f64>,
    rm_mean_slopes: bool,
    #[serde(rename = "_Tu")]
    tu: Vec<f64>,
    rho_3: f64,
    end2end_ordering: bool,
}
#[derive(Deserialize)]
struct QP {
    #[serde(rename = "SHAcO_qp")]
    data: QPData,
}

// wfs48x48 sample structure
#[derive(Deserialize)]
struct WFSData {
    wfsdata: Vec<f64>,
}

// Number of bending modes (it can be retrieved from D)
const N_BM: u8 = 27;

fn main() {
    // Import AcO data
    let qp: QP = {
        let file = File::open("SHAcO_qp_rhoP1e-3_kIp5.rs.pkl").unwrap();
        let rdr = BufReader::with_capacity(10_000, file);
        pickle::from_reader(rdr).unwrap()
    };
    println!("Number of lenslets:{}", qp.data.wfs_mask[0].len());

    // Compute QP quadratic term matrix
    let w2 = Matrix_ncxnc::from_vec(qp.data.w2);
    let w3 = Matrix_ncxnc::from_vec(qp.data.w3);
    let ns = qp.data.dmat.len()/271;
    println!("Valid lenslets:{}", qp.data.dmat.len()/271);
    let d_wfs = DynMatrix::from_vec(271, 7360, qp.data.dmat).transpose();

    let dT_w1_d = {
        //let dT_w1_d_dyn = d_wfs.clone_owned().transpose() * d_wfs.clone_owned();
        let dT_w1_d_dyn = d_wfs.tr_mul(&d_wfs);
        Matrix_ncxnc::from_vec(dT_w1_d_dyn.as_slice().to_vec())
    };

    // Extract the upper triangular elements of `P`
    let p_utri = {
        println!("rho_3:{}", qp.data.rho_3);
        let rho_3_ksq = qp.data.rho_3*qp.data.k*qp.data.k;
        let p = dT_w1_d + w2 + (w3.scale(rho_3_ksq));

        CscMatrix::from_column_iter_dense(
            p.nrows(),p.ncols(),p.as_slice().to_vec().into_iter())
            .into_upper_tri()
    };
    // Linear QP problem term
    // Import AcO data
    let q_struct: WFSData = {
        let file = File::open("wfs48x48sample.rs.pkl").unwrap();
        let rdr = BufReader::with_capacity(100_000, file);
        pickle::from_reader(rdr).unwrap()
    };
    println!("Number of lenslets:{}", q_struct.wfsdata.len());
    
    let mut y = Vec::new();//with_capacity(7360);
    for seg_mask in qp.data.wfs_mask.iter(){
        // Commands to take indices of true elements:
        // https://codereview.stackexchange.com/questions/159652/indices-of-true-values
        let valid_l_idxs: Vec<_> = seg_mask.iter()
            .enumerate()
            .filter(|&(_, &value)| value)
            .map(|(index, _)| index)
            .collect();
        println!("...{}", valid_l_idxs.len());

        for iv in valid_l_idxs {
            y.push(q_struct.wfsdata[iv]);
        }

        // ::retain may be an alternative
        // https://doc.rust-lang.org/std/vec/struct.Vec.html#method.retain
    }
    // WFS meas dimension check
    assert_eq!(ns, y.len());

    let q : Vec<f64> = {
        let y_vec = -Vector_ns::from_vec(y);
        (d_wfs.transpose() * y_vec).as_slice().to_vec()
    };
    assert_eq!(271, q.len());
    //q = -y_valid.T.dot(self.W1_D) - self.rho3*u_ant.T.dot(self.W3)*self.k_I

    // Indices to insert (or remove) S7Rz columns
    let iM1S7Rz: u8 =
        if qp.data.end2end_ordering {41}
        else {((12+N_BM)*6) + 5};
    let iM2S7Rz: u8 =   // Add 1 to delete
        if qp.data.end2end_ordering {82+1}
        else  {((12+N_BM)*6) + 10};
    
    // Inequality constraint matrix: lb <= a_in*u <= ub
    let a_in = {
        let tu = DynMatrix::from_vec(273, 1228, qp.data.tu).transpose()
            .remove_columns_at(&[iM1S7Rz.into(),iM2S7Rz.into()]);
        // Remove S7Rz from Tu
        println!("Number of Tu cols:{}", tu.ncols());

        CscMatrix::from_column_iter_dense(
            tu.nrows(), tu.ncols(),
            tu.scale(qp.data.k).as_slice().to_vec().into_iter())
    };
    
    // QP settings
    let settings = Settings::default()
        .eps_abs(1.0e-8)
        .eps_rel(1.0e-6)
        .max_iter(500*271)
        .warm_start(true)
        .verbose(true);
     
    // Create an OSQP problem
    let mut prob = Problem::new(p_utri,&q,a_in,&qp.data.umin,&qp.data.umax,&settings)
       .expect("Failed to setup problem!");

    // Solve problem
    let result = prob.solve();
    
    // Print the solution
    let u = result.x().expect("Failed to solve problem!");
    for i in 0..7 {
        println!("{}",format!("{:.4e}", u[i]));
    }
}
