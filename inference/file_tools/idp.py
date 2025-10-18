import os 
import json

from alibabacloud_docmind_api20220711.client import Client as docmind_api20220711Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_docmind_api20220711 import models as docmind_api20220711_models
from alibabacloud_tea_util.client import Client as UtilClient
from alibabacloud_tea_util import models as util_models
from alibabacloud_credentials.client import Client as CredClient

key = os.environ.get('IDP_KEY_ID')
secret = os.environ.get('IDP_KEY_SECRET')


class IDP():
    def __init__(self):
        config = open_api_models.Config(
            access_key_id=key,
            access_key_secret=secret
        )
        config.endpoint = f'docmind-api.cn-hangzhou.aliyuncs.com'
        self.client = docmind_api20220711Client(config)

    def file_submit_with_url(self, file_url):
        print('parsing with document url ', file_url)
        file_name = os.path.basename(file_url)
        request = docmind_api20220711_models.SubmitDocParserJobAdvanceRequest(
            file_url=file_url,
            file_name=file_name,
            reveal_markdown=True,
    )
        runtime = util_models.RuntimeOptions()
        result_dict = None
        try:
            response = self.client.submit_doc_parser_job_advance(request,runtime)
            result_dict = response.body.data.id
        except Exception as error:
            UtilClient.assert_as_string(error.message)  

        return result_dict


    def file_submit_with_path(self, file_path):
        print('parsing with document local path ', file_path)
        file_name = os.path.basename(file_path)
        request = docmind_api20220711_models.SubmitDocParserJobAdvanceRequest(
            file_url_object=open(file_path, "rb"),
            file_name=file_name,
    )
        runtime = util_models.RuntimeOptions()
        result_dict = None
        try:
            response = self.client.submit_doc_parser_job_advance(request, runtime)
            result_dict = response.body.data.id
        except Exception as error:
            print(error)
            UtilClient.assert_as_string(error.message)

        return result_dict

    def file_parser_query(self,fid):
        request = docmind_api20220711_models.QueryDocParserStatusRequest(
            id=fid
        )
        try:
            response = self.client.query_doc_parser_status(request)
            NumberOfSuccessfulParsing = response.body.data
        except Exception as e:
            print(e)
            return None
        status_parse = response.body.data.status
        NumberOfSuccessfulParsing = NumberOfSuccessfulParsing.__dict__
        responses = dict()
        for i in range(0, NumberOfSuccessfulParsing["number_of_successful_parsing"], 3000):
            request = docmind_api20220711_models.GetDocParserResultRequest(
                id=fid,
                layout_step_size=3000,
                layout_num=i
            )
            try:
                response = self.client.get_doc_parser_result(request)
                result = response.body.data
                if not responses:
                    responses = result
                else:
                    responses['layouts'].extend(result['layouts'])
            except Exception as error:
                return None,status_parse
        return responses,status_parse
  	