import json
from typing import Dict, Any

from allennlp.common import JsonDict
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('iterative_template_extraction_famus')
class IterativeTemplateExtractionPredictorFAMuS(Predictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        if 'predicted_bpjson_template_dict' in outputs:
            return json.dumps(outputs['predicted_bpjson_template_dict']) + "\n"
        elif 'predicted_muc_template_dict' in outputs:
            return json.dumps(outputs['predicted_muc_template_dict']) + "\n"
        elif 'predicted_scirex_template_dict' in outputs:
            return json.dumps(outputs['predicted_scirex_template_dict']) + "\n"
        else:
            raise ValueError("No templates to write!")

    def _to_params(self) -> Dict[str, Any]:
        return super()._to_params()
