// -*- C++ -*-
#include <iostream>
#include <fstream>
#include <vector>
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/IdentifiedFinalState.hh"
#include "Rivet/Projections/VetoedFinalState.hh"
#include "Rivet/Projections/PromptFinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/DressedLeptons.hh"


namespace VBFZ
{
  struct Observables
  {
    double m_ll, pT_ll, theta_ll, rap_ll ;
    double m_jj, pT_jj, theta_jj, rap_jj ;
    double pT_j1, pT_j2, Dy_j_j, Dphi_j_j ;
    int    N_jets, N_gap_jets ;
    double weight ;

    Observables (const double & w = 1.0) :
      m_ll       (-99.) ,
      pT_ll      (-99.) ,
      theta_ll   (-99.) ,
      rap_ll     (-99.) ,
      m_jj       (-99.) ,
      pT_jj      (-99.) ,
      theta_jj   (-99.) ,
      rap_jj     (-99.) ,
      pT_j1      (-99.) ,
      pT_j2      (-99.) ,
      Dy_j_j     (-99.) ,
      Dphi_j_j   (-99.) ,
      N_jets     (-99 ) ,
      N_gap_jets (-99 ) ,
      weight     (w   )
    {
    }

    void print (std::ofstream & file)
    {
      file << "  [" << weight << ",  " << m_ll   << ",  " << pT_ll    << ",  " << theta_ll << ",  " << rap_ll ;
      file << ",  " << m_jj   << ",  " << pT_jj  << ",  " << theta_jj << ",  " << rap_jj   << ",  " << pT_j1 ;
      file << ",  " << pT_j2  << ",  " << Dy_j_j << ",  " << Dphi_j_j << ",  " << N_jets   << ",  " << N_gap_jets << "]" << std::endl ;
    }

  } ;

  int getNumGapJets (const std::vector<Rivet::Jet>& jets) 
  { 
    if (jets.size() < 3) return 0 ;
    int num_gap_jets (0) ;
    double y1 (jets[0].rapidity()), y2 (jets[1].rapidity()) ;
    if ( y1 < y2 ) std::swap(y1, y2) ;
    for (unsigned int i (2); i < jets.size(); ++i)
    {
      const Rivet::Jet& j (jets[i]) ;
      double jet_rap (j.rapidity()) ;
      if ( jet_rap > y1 ) continue ;
      if ( jet_rap < y2 ) continue ;
      num_gap_jets += 1 ;
    }
    return num_gap_jets ;
  }
}

namespace Rivet {

  /// VBFZ in pp at 13 TeV
  class event_parser : public Analysis 
  {
  protected:

    ofstream * _outFile ;

  public:

    /// Constructor
    event_parser(string name="event_parser")
      : Analysis(name), _outFile(NULL)
    {
    }

    /// Book histograms and initialise projections before the run
    void init()
    {
      FinalState fs(-5.0, 5.0);

      IdentifiedFinalState photon_fs(fs);
      photon_fs.acceptIdPair(PID::PHOTON);
      PromptFinalState photons(photon_fs);

      IdentifiedFinalState el_id(fs);
      el_id.acceptIdPair(PID::ELECTRON);
      PromptFinalState electrons(el_id);

      IdentifiedFinalState mu_id(fs);
      mu_id.acceptIdPair(PID::MUON);
      PromptFinalState muons(mu_id);

      Cut cuts_el = (Cuts::pT > 25*GeV) && (Cuts::abseta < 1.37 || (Cuts::abseta > 1.52 && Cuts::abseta < 2.47));
      Cut cuts_mu = (Cuts::pT > 25*GeV) && (Cuts::abseta < 2.4);

      DressedLeptons dressed_electrons(photons, electrons, 0.1, cuts_el);
      declare(dressed_electrons, "DressedElectrons");

      DressedLeptons dressed_muons(photons, muons, 0.1, cuts_mu);
      declare(dressed_muons, "DressedMuons");

      FastJets jets(fs, FastJets::ANTIKT, 0.4, JetAlg::NO_MUONS, JetAlg::NO_INVISIBLES);
      declare(jets, "Jets");

      _outFile = new std::ofstream("event_parser_output.dat") ;
      (*_outFile) << "[DATA]" << std::endl ;
      (*_outFile) << "Keys: ['float::weight',  'float::m_ll',  'float::pT_ll',  'float::theta_ll',  'float::rap_ll',  'float::m_jj',  'float::pT_jj',  'float::theta_jj',  'float::rap_jj',  'float::pT_j1',  'float::pT_j2',  'float::Dy_j_j',  'float::Dphi_j_j',  'int::N_jets',  'int::N_gap_jets']" << std::endl ;
      (*_outFile) << "Events:" ;
    }

    /// Perform the per-event analysis
    void analyze(const Event& event)
    {
      // Initialise all observables
      VBFZ::Observables observables (event.weight()) ;

      // access fiducial electrons and muons
      const Particle *l1 (nullptr), *l2 (nullptr) ;
      auto muons     = apply<DressedLeptons>(event, "DressedMuons"    ).dressedLeptons() ;
      auto electrons = apply<DressedLeptons>(event, "DressedElectrons").dressedLeptons() ;

      // Dilepton selection: 2 OSSF leptons
      if ( muons.size() >= 2 )
      {
        l1 = &muons[0]; 
        l2 = &muons[1]; 
      }
      else if ( electrons.size() >= 2 )
      {
        l1 = &electrons[0]; 
        l2 = &electrons[1]; 
      }
      else
      {
        observables.print(*_outFile) ; 
        vetoEvent ;
      }
      if ( l1->threeCharge() + l2->threeCharge() != 0 )
      {
        observables.print(*_outFile) ; 
        vetoEvent ;
      }

      // create dilepton system
      FourMomentum dilepon_system (l1->mom() + l2->mom()) ;
      observables.m_ll     = dilepon_system.mass () / GeV ;
      observables.pT_ll    = dilepon_system.pt   () / GeV ;
      observables.theta_ll = dilepon_system.theta() ;
      observables.rap_ll   = dilepon_system.absrapidity() ;

      // Electron-jet overlap removal
      std::vector<Jet> jets;
      auto all_jets = apply<FastJets>(event, "Jets").jetsByPt(Cuts::pT > 30.0*GeV && Cuts::absrap < 4.4);
      for (const auto & j: all_jets) {
         bool near_electron (false) ;
         for (const auto & e: electrons)
         {
           if ( deltaR(j.mom(), e.mom(), RAPIDITY) > 0.2 ) continue ;
           near_electron = true ;
         }
         if (near_electron) continue ;
	       jets.push_back(j);
      }
      observables.N_jets = jets.size() ;

      // If no surviving jets, stop processing event
      if ( jets.size() == 0 )
      {
        observables.print(*_outFile) ; 
        vetoEvent;
      }

      // Fill one jet observables
      Jet & j1 = jets [0] ;
      observables.pT_j1 = j1.pt() ;

      // Dijet system requires
      if ( jets.size() < 2 )
      {
        observables.print(*_outFile) ; 
        vetoEvent;
      }

      // Fill dijet observables
      Jet & j2 = jets [1] ;
      FourMomentum dijet_system (j1.mom() + j2.mom()) ;
      observables.pT_j2      = j2.pt() / GeV ;
      observables.Dphi_j_j   = ((j1.rapidity() > j2.rapidity()) ? mapAngleMPiToPi(j1.phi() - j2.phi()) : mapAngleMPiToPi(j2.phi() - j1.phi())) ;
      observables.Dy_j_j     = fabs(deltaRap(j1.mom(), j2.mom())) ;
      observables.m_jj       = dijet_system.mass() / GeV ;
      observables.pT_jj      = dijet_system.pt() / GeV ;
      observables.theta_jj   = dijet_system.theta() ;
      observables.rap_jj     = dijet_system.absrapidity() ;
      observables.N_gap_jets = VBFZ::getNumGapJets(jets) ;

      // Print full observables
      observables.print(*_outFile) ; 
    }

    void finalize()
    {
      (*_outFile) << "Xsec_per_event: " << crossSectionPerEvent() << std::endl ;
      _outFile -> close() ;
      delete _outFile ;
      _outFile = NULL ;
    }

  };

  DECLARE_RIVET_PLUGIN(event_parser);
}
