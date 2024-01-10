//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//


#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include <iostream>
#include <cmath>


namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/test/anymal_b_simple_description/robots/anymal-kinova-collision.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
//    std::cout << gvDim_ << std::endl;
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_),prevTarget_.setZero(gcDim_),prevPrevTarget_.setZero(gcDim_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8,0.0, 2.62, -1.57, 0.0, 2.62, 0.0;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);jointPgain.segment(6,12).setConstant(100.0);
//          std::cout << jointPgain.transpose() << std::endl;
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2); jointDgain.segment(6,12).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 52;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("kinova_link_6"));

      auto RR_footIndex = anymal_->getBodyIdx("LF_SHANK");
      auto RL_footIndex = anymal_->getBodyIdx("RF_SHANK");
      auto FR_footIndex = anymal_->getBodyIdx("LH_SHANK");
      auto FL_footIndex = anymal_->getBodyIdx("RH_SHANK");

      true_contact.setZero();

      //      for (const auto& element : footIndices_) {
//          std::cout << ' ' << element;
//      }
//      std::cout << std::endl;
//    EEFrameIndex_.insert(anymal->getBodyIdx("j2s6s200_link_finger_tip_2"));
//      auto RR_footIndex = robot_->getBodyIdx("RR_calf");
//      auto RL_footIndex = robot_->getBodyIdx("RL_calf");
//      auto FR_footIndex = robot_->getBodyIdx("FR_calf");
//      auto FL_footIndex = robot_->getBodyIdx("FL_calf");

      posError_.setZero();
      TEEpos_.setZero();
      PEEpos_.setZero();

      /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(anymal_);
      visual_target = server_->addVisualSphere("visual_target",0.05,1,0,0,0.4);
//      visual_target2 = server_->addVisualSphere("visual_target2",0.05,0,1,0,0.4);
//        visual_target2 = server_->addVisualBox("visual_target2",0.1,0.1,0.1,0,1,0,0.4);
        visual_EEpos = server_->addVisualSphere("visual_EEpos",0.05,0,0,1,0.4);
    }
  }

  void init() final { }


  void reset() final {
    anymal_->setState(gc_init_, gv_init_);
    updateObservation();
    if (visualizable_) {
      Eigen::Vector3d des_pos(0.2*uniDist_(gen_)+2.0,0.2*uniDist_(gen_)-1.5,0.2*uniDist_(gen_)+0.6);
      visual_target->setPosition(des_pos);
      TEEpos_ = visual_target->getPosition();
    }
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    prevPrevTarget_=  prevTarget_;
    prevTarget_ = pTarget_;
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
//    pTarget_[22] = gc_init_[22];
//    pTarget_[24] = gc_init_[24];

//    pTarget_.segment(7,12) = gc_init_.segment(7,12);
    anymal_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();

      if (visualizable_) {
//          visual_target2 ->setPosition(TEEpos_);
          visual_EEpos->setPosition(PEEpos_.e());
      }
      Eigen::VectorXd jointPosTemp(12), jointPosWeight(12);
      jointPosWeight << 1.0, 0.,0.,1.,0.,0.,1.,0.,0.,1.,0.,0.;
      jointPosTemp = gc_.segment(7,12) - gc_init_.segment(7,12);
      jointPosTemp = jointPosWeight.cwiseProduct(jointPosTemp.eval());

//      rewards_.record("footSlip", footSlip_.sum());
      rewards_.record("EEpos", std::exp(-posError_.norm()));
//      rewards_.record("Joint2", std::exp(-(gc_[20]-2.62)*(gc_[2]-2.62)));
      rewards_.record("Height", std::exp(-(gc_[2]-0.46)*(gc_[2]-0.46)));
//      rewards_.record("bodyOri", std::acos(bodyOri_) * std::acos(bodyOri_));

      rewards_.record("Lsmoothness1",(pTarget_.segment(7,12) - prevTarget_.segment(7,12)).squaredNorm());
      rewards_.record("Jsmoothness1",(pTarget_.tail(6) - prevTarget_.tail(6)).squaredNorm());
      rewards_.record("smoothness2", (pTarget_ - 2 * prevTarget_ + prevPrevTarget_).squaredNorm());
      rewards_.record("jointPos", jointPosTemp.squaredNorm());
      rewards_.record("pTarget", (pTarget_-actionMean_).squaredNorm());
      rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
//    rewards_.record("bodyLinearVel", bodyLinearVel_.norm());
//    rewards_.record("bodyAngularVel", bodyAngularVel_.norm());


//    rewards_.record("torque", Tor.squaredNorm());


//    rewards_.record("forwardvel", std::min(4.0,bodyLinearVel_[0]));
//    std::cout <<"T1: "<<std::exp(-posError_.norm())<<std::endl;
//    std::cout <<"T1: "<<std::min(4.0,bodyLinearVel_[0])<<std::endl;

//    std::cout << posError_.norm() << std::endl;

    return rewards_.sum();
  }

  void updateObservation() {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
    bodyOri_ = rot.e()(8);

    raisim::Mat<3,3> PEErot_;
    Eigen::Vector3d PEEori_;

    auto EEFrameIndex_ = anymal_->getFrameIdxByName("kinova_joint_end_effector");
    anymal_->getFramePosition(EEFrameIndex_, PEEpos_);
    anymal_->getFrameOrientation(EEFrameIndex_,PEErot_);
    posError_ = TEEpos_-PEEpos_.e();
    PEEori_ = PEErot_.e().col(2);
    Eigen::Vector3d posError = rot.e().transpose() * (posError_);
//    std::cout << posError << std::endl;


      obDouble_ << gc_[2], /// body height : 1
        rot.e().row(2).transpose(), /// body orientation : 3
        gc_.tail(18), /// joint angles : 12
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity : 6
        gv_.tail(18),
        posError,
        rot.e().transpose()* PEEori_;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;

    terminalReward = 0.f;

    return false;
  }

  void curriculumUpdate() { };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* anymal_;
  raisim::Visuals* visual_target;
  raisim::Visuals* visual_EEpos;

  raisim::Visuals* visual_target2;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_,prevTarget_,prevPrevTarget_, vTarget_,true_contact;
  double terminalRewardCoeff_ = -10.,bodyOri_;
  raisim::Vec<3> PEEpos_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, TEEpos_, posError_;
  std::set<size_t> footIndices_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  thread_local static std::mt19937 gen_;
  thread_local static std::normal_distribution<double> normDist_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
};
thread_local std::mt19937  raisim::ENVIRONMENT::gen_;
thread_local std::normal_distribution<double> raisim::ENVIRONMENT::normDist_(0., 1.);
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(-1., 1.);
}
