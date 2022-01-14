//
// Active optics (AcO) validation test script
// 
// In this script we consider the AcO interaction matrix 
// as the SH-WFS model. The library file lib.rs implements
// the active optics algorithm (reconstructor & controller).
//
// January 2022
//

use aco_loop_example::QP;
use dosio::{ios, Dos};
use nalgebra::{DVector};

//use serde::Deserialize;
//use serde_pickle as pickle;
//use std::{fs::File, io::BufReader};

const N_MODE: usize = 271;
const M1_BM: usize = 27;
const M1_RBM: usize = 41;
const M2_RBM: usize = 41;
const I_M1_S7_RZ: usize = M1_RBM;
const I_M2_S7_RZ: usize = M1_RBM + M2_RBM;
const N_LOOP: usize = 8;


fn main() -> Result<(),()> {

    println!("AcO debug test ...");

    // Active optics control algorithm
    let mut aco = QP::<M1_RBM, M2_RBM, 27, N_MODE>::new(
        "../aco_impl_stdalone/SHAcO_qp_rhoP1e-3_kIp5.rs.pkl")
        .unwrap()
        .build();
  
    // Get AcO interaction matrix (Dmatrix: ns x nc)
    let d_wfs = aco.get_d_wfs();
    
    //let filename = File::open("../aco_poke.pkl").unwrap();
    //let _buf: Vec<f64> = pkl::from_reader(
    //    filename,pkl::de::DeOptions::new()).unwrap();

    //let file = File::open("aco_poke.pkl")?;
    //let rdr = BufReader::with_capacity(10_000, file);
    //let Drust: Self = pickle::from_reader(rdr, Default::default())?;

    // Misalignment & Figure error
    let mut m1_rbm_buf = vec![vec![0f64; 6]; 7];
    m1_rbm_buf[0][0] = 1.1e-6; // M1S1-Tx: 
    m1_rbm_buf[1][1] = 1.2e-6; // M1S2-Ty:
    m1_rbm_buf[2][3] = 1.4e-6; // M1S3-Rx:
    m1_rbm_buf[3][4] = 1.5e-6; // M1S4-Ry: 
    m1_rbm_buf[4][2] = 1.6e-6; // M1S5-Tz:
    m1_rbm_buf[5][5] = 1.3e-6; // M1S5-Rz: 
    m1_rbm_buf[6][5] = 2e-6; // M1S7-Rz: 
    let m1_rbm = m1_rbm_buf.into_iter().flatten().collect::<Vec<f64>>();
    let mut m2_rbm_buf = vec![0f64; 42];
    m2_rbm_buf[M2_RBM] = 0. * 3e-4; // M2S7-Rz
    let m2_rbm = m2_rbm_buf;
    let mut m1_modes_buf = vec![vec![0f64; M1_BM]; 7];
    m1_modes_buf[0][0] = 4e-6;
    m1_modes_buf[0][2] = 5e-6;    
    let m1_modes = m1_modes_buf.into_iter().flatten().collect::<Vec<f64>>();

    let d: Vec<f64> = [m1_rbm,m2_rbm,m1_modes].concat();

    //Initialize AcO command vector
    let mut u = vec![0f64; N_MODE+2];

    // AcO loop
    for k_ in 0..N_LOOP {

        let mut state: Vec<f64> = d.iter()
            .zip(u.iter())
            .map(|(&v, &w)| v + w)
            .collect();

        //let state = d + u;
        state.remove(I_M1_S7_RZ);
        state.remove(I_M2_S7_RZ);

        //let y_valid = gosm_aco.in_step_out(Some(vec![m1_rbm, m2_rbm, m1_modes]))?;
        let y_valid = vec![ios!(SensorData(
            (&d_wfs * &DVector::from_column_slice(&state))
            .as_slice()
            .to_vec()
        ))];

        //Reconstruction and integration
        let mut sol_aco = aco.in_step_out(Some(y_valid))
            .unwrap()
            .unwrap();

        //Unwrap AcO data
        let m2_pos_cmd = vec![sol_aco.pop().unwrap()];
        let u_m2rbm = Option::<Vec<f64>>::from(&m2_pos_cmd[ios!(M2poscmd)]).unwrap();
        let m1_rbm_cmd = sol_aco.pop().map(|x| vec![x]).unwrap();
        let u_m1rbm = Option::<Vec<f64>>::from(&m1_rbm_cmd[ios!(M1RBMcmd)]).unwrap();
        
        let m1s1bm = Option::<Vec<f64>>::from(&sol_aco[ios!(M1S1BMcmd)]).unwrap();
        let m1s2bm = Option::<Vec<f64>>::from(&sol_aco[ios!(M1S2BMcmd)]).unwrap();
        let m1s3bm = Option::<Vec<f64>>::from(&sol_aco[ios!(M1S3BMcmd)]).unwrap();
        let m1s4bm = Option::<Vec<f64>>::from(&sol_aco[ios!(M1S4BMcmd)]).unwrap();
        let m1s5bm = Option::<Vec<f64>>::from(&sol_aco[ios!(M1S5BMcmd)]).unwrap();
        let m1s6bm = Option::<Vec<f64>>::from(&sol_aco[ios!(M1S6BMcmd)]).unwrap();
        let m1s7bm = Option::<Vec<f64>>::from(&sol_aco[ios!(M1S7BMcmd)]).unwrap();
        
        u = [u_m1rbm,u_m2rbm,
            m1s1bm,m1s2bm,m1s3bm,m1s4bm,m1s5bm,m1s6bm,m1s7bm].concat();

        println!("Iter {} of {} -> AcO output: {:1.2?}\n",
            k_+1, N_LOOP,
            u.iter().map(|x| 1e6 * x).collect::<Vec<f64>>());

    }

    println!("Optical DOF static disturbance: {:?}",
        d.iter().map(|x| 1e6 * x).collect::<Vec<f64>>());
    Ok(())

}

