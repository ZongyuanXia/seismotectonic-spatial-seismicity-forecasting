"""Small checks for the Japan adapter; no fitting or target-driven tuning."""
import argparse, json, sys, unittest
from pathlib import Path
import numpy as np
import pandas as pd
import predict_etas_trial as p
from etas.inversion import upper_gamma_ext

ROOT=None
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.region=json.loads((p.BASE/'region_meta.json').read_text())
    def test_grid_matches_all_frozen_event_keys(self):
        data=ROOT/'datasets/cl10_mapping_v2'
        if not data.exists():data=ROOT/'cl10_mapping_v2'
        audit=pd.read_csv(data/'usgs_event_audit.csv')
        keys,ok=p.grid_keys(audit.latitude,audit.longitude,self.region)
        self.assertTrue(ok.all());np.testing.assert_array_equal(keys,audit.key)
    def test_geographic_cell_areas_and_budget(self):
        area=p.areas(self.region);self.assertTrue((area>0).all())
        self.assertGreater(area[0,0],area[-1,0])
        score=np.random.default_rng(1).random((340,200))
        for fraction in (.005,.01,.02,.03):
            alarm=p.ranked_alarm(score,area,fraction)
            self.assertLessEqual(abs(area[alarm].sum()-area.sum()*fraction),area.max())
    def test_tie_order_and_area_reproducible(self):
        area=p.areas(self.region);score=np.zeros((340,200))
        np.testing.assert_array_equal(p.ranked_alarm(score,area,.01),p.ranked_alarm(score,area,.01))
    def test_background_sampler_seed_and_boundary(self):
        b=self.region['effective_bounds'];p.CTX['bounds']=(b['lat_min'],b['lon_min'],b['lat_max'],b['lon_max'])
        lat=pd.Series([33.,37.,42.]);lon=pd.Series([133.,138.,142.]);weights=pd.Series([.1,.3,.6])
        np.random.seed(7);a=p.exact_background(lat,lon,weights,n=500)
        np.random.seed(7);c=p.exact_background(lat,lon,weights,n=500)
        np.testing.assert_array_equal(a,c)
        self.assertTrue(((a[0]>=b['lat_min'])&(a[0]<b['lat_max'])).all())
        self.assertTrue(((a[1]>=b['lon_min'])&(a[1]<b['lon_max'])).all())
    def test_exact_time_inverse_matches_author_survival(self):
        c=.01;tau=4000.;omega=-.13
        np.random.seed(9);u=np.random.uniform(size=1000)
        np.random.seed(9);t=p.simulation.simulate_aftershock_time(np.log10(c),omega,np.log10(tau),1000)
        self.assertTrue((t>=0).all())
        recovered=1-upper_gamma_ext(-omega,(t+c)/tau)/upper_gamma_ext(-omega,c/tau)
        np.testing.assert_allclose(u,recovered,rtol=1e-9,atol=1e-10)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    a,rest=ap.parse_known_args();ROOT=a.root
    unittest.main(argv=[sys.argv[0],*rest],verbosity=2)
