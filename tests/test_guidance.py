"""Behavioral checks for spatial guidance; no weights or network required."""
import unittest
import torch
from focusedit.guidance import FocusEditModel
from focusedit._backbone.model import NaviFlowAlignModel


class ToyBaseline(NaviFlowAlignModel):
    def _predict_vector(self, x, t, embed):
        return 0.25*x + embed[0]


class ToyFocus(FocusEditModel):
    def _predict_vector(self, x, t, embed):
        return 0.25*x + embed[0]


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"cfg_scale": 3.0, "noise_samples": 1}
        self.x = torch.zeros(1,2,4,4)
        self.source = (torch.tensor(0.0), torch.tensor(0.0))
        self.target = (torch.tensor(1.0), torch.tensor(0.0))

    def run_update(self, model, state=None):
        torch.manual_seed(123)
        return model._estimate_updates(self.x, self.x if state is None else state,
                                       torch.tensor(500.),torch.tensor(.5),self.source,self.target)

    def test_all_one_weights_match_baseline(self):
        base=ToyBaseline(None,None,self.cfg)
        focus=ToyFocus(None,None,self.cfg)
        focus.set_edit_mask(torch.ones(1,1,4,4))
        for a,b in zip(self.run_update(base),self.run_update(focus)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_spatial_support_only_scales_guidance(self):
        focus=ToyFocus(None,None,self.cfg)
        weight=torch.full((1,1,4,4),.2); weight[:,:,:,2:]=1
        focus.set_edit_mask(weight)
        flow,_=self.run_update(focus)
        torch.testing.assert_close(flow,3*weight.expand_as(flow))

    def test_zero_guidance_can_leave_source_residual(self):
        focus=ToyFocus(None,None,self.cfg)
        focus.set_edit_mask(torch.zeros(1,1,4,4))
        flow,_=self.run_update(focus,state=torch.ones_like(self.x))
        torch.testing.assert_close(flow,torch.full_like(self.x,.25))


if __name__ == "__main__":
    unittest.main()
