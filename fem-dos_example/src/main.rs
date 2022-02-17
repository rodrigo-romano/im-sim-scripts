//
//
//

use fem::{
    dos::{DiscreteModalSolver, Exponential},
    FEM
};
use dosio::{ios, Dos, IOVec};
// ios macro provides the argument to append the list of model IOs
// Dos: required for mnt_drives.in_step_out()
// IOVec: required for fem_outputs.pop_these()
use mount_ctrl as mount;
use serde_pickle as pkl;
use std::fs::File;
use std::error::Error;

// Constants
const N_STEP: i32 = 10; //1_000;


fn main() -> Result<(), Box<dyn Error>> {
    // MOUNT CONTROL
    let mut mnt_drives = mount::drives::Controller::new();
    let mut mnt_ctrl = mount::controller::Controller::new();

    // Import GMT structural dynamics model
    // let fem = FEM::from_env()?;
    let mut fem = DiscreteModalSolver::<Exponential>::from_fem(FEM::from_env()?)
        .sampling(1e3)
        .proportional_damping(2. / 100.)
        .inputs_from(&[&mnt_drives])
        .inputs(ios!(OSSM1Lcl6F))
        .outputs(ios!(OSSM1Lcl, MCM2Lcl6D))
        .outputs(ios!(
            OSSAzEncoderAngle,
            OSSElEncoderAngle,
            OSSRotEncoderAngle
    ))
    .build()?;
    //println!("FEM eigen frequencies: {:?}", &fem.state_space.eigen_frequencies[..5]);
    println!("{}", fem);


    // Initialize FEM outputs
    let mut fem_outputs = fem.zeroed_outputs();
    
    let mut m1_logs = Vec::<Vec<f64>>::with_capacity((N_STEP * 42).try_into().unwrap());
    let mut m2_logs = Vec::<Vec<f64>>::with_capacity((N_STEP * 42).try_into().unwrap());

    for ii in 0..N_STEP {
        //let mut fem_forces = vec![ios!(OSSM1Lcl6F(vec![0f64; 42]))];
        let mut fem_forces = vec![ios!(OSSM1Lcl6F(vec![0f64; 42]))];
        // println!("1){}\n", fem_forces.len());
        if ii<4 {
            println!("{:?}\n", &fem_forces);
        }

        fem_outputs
            .pop_these(ios!(
                OSSElEncoderAngle,
                OSSAzEncoderAngle,
                OSSRotEncoderAngle
            ))
            .and_then(|mut mnt_encdr| {
                mnt_ctrl
                    .in_step_out(Some(mnt_encdr.clone()))
                    .unwrap()
                    .and_then(|mut mnt_cmd| {
                        mnt_cmd.append(&mut mnt_encdr);
                        mnt_drives.in_step_out(Some(mnt_cmd)).unwrap()
                    })
            })
            .map(|mut mount_drives_forces| {
                fem_forces.append(&mut mount_drives_forces);
            });

            //println!("2){}\n", fem_forces.len());
            if ii< 4 {
                println!("{:?}\n", &fem_forces);
            }
            fem_outputs = fem.in_step_out(Some(fem_forces)).unwrap().unwrap();

        Option::<Vec<f64>>::from(&fem_outputs[ios!(OSSM1Lcl)]).map(|x| m1_logs.push(x));
        Option::<Vec<f64>>::from(&fem_outputs[ios!(MCM2Lcl6D)]).map(|x| m2_logs.push(x));
    }

    let mut file = File::create("test_data.pkl").unwrap();
    pkl::to_writer(&mut file, &(m1_logs, m2_logs), true).unwrap();

    Ok(())
}

/*

        
*/
