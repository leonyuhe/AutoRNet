import logging
import re
from openai import OpenAI
import json
from typing import List, Dict
import concurrent.futures
from AutoRNet import Util

# 定义一个可序列化的响应函数
def get_response(api_key, base_url, model, prompt_content) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    json_content = ''
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You're an algorithm expert and a helpful assistant"},
                {"role": "user", "content": prompt_content},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )

        response_content = response.choices[0].message.content
        response_content = re.sub(r'^\s*```json\s*|\s*```\s*$', '', response_content, flags=re.DOTALL)
        response_content = re.sub(r'(?<!\\)\\(?![nrtbf\\"\'/u])', r'\\\\', response_content)
        response_content = re.sub(r'"""\s*(.*?)\s*"""', lambda m: '"' + m.group(1).replace('\n', '\\n').replace('"', '\\"') + '"', response_content, flags=re.DOTALL)

        json2dict_content = json.loads(response_content)
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e} response_content: {response_content}")
        raise
    return json2dict_content

class Connector:
    def __init__(self, max_workers=4, **kwargs):
        try:
            self.model = kwargs['MODEL']
            self.api_key = kwargs['CONNECT_API_KEY']
            self.base_url = kwargs['CONNECT_BASE_URL']
            self.max_workers = max_workers
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            raise
    def get_response(self, prompt_content):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response_content =''
        try:
            response = client.chat.completions.create(
                model= self.model,
                messages=[
                    {"role": "system", "content": "You're an algorithm expert and a helpful assistant"},
                    {"role": "user", "content": prompt_content},
                ],
                max_tokens=1024,
                temperature=0.7,
                stream=False
            )

            response_content =response.choices[0].message.content
            response_content = re.sub(r'^\s*```json\s*|\s*```\s*$', '', response_content, flags=re.DOTALL)
            response_content = re.sub(r'(?<!\\)\\(?![nrtbf\\"\'/u])', r'\\\\', response_content)
            response_content = re.sub(r'"""\s*(.*?)\s*"""', lambda m: '"' + m.group(1).replace('\n', '\\n').replace('"', '\\"') + '"', response_content, flags=re.DOTALL)

            json2dict_content = json.loads(response_content)
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e} response_content: {response_content}")
            raise
        return json2dict_content
    def get_concurrent_response(self, prompts: List[str], max_retries: int = 3) -> List[str]:
        retries = 0
        results: Dict[int, str] = {}
        failed_indices = set(range(len(prompts)))

        while retries < max_retries and failed_indices:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                try:
                    future_to_index = {
                        executor.submit(get_response, self.api_key, self.base_url, self.model, prompts[i]): i
                        for i in failed_indices
                    }
                    failed_indices.clear()
                    for future in concurrent.futures.as_completed(future_to_index, timeout=Util.http_concurrent_timeout):
                        index = future_to_index[future]
                        try:
                            result = future.result(timeout=Util.concurrent_timeout)
                            results[index] = result
                        except Exception as exc:
                            logging.error(f'get_concurrent_response at index {index} generated an exception: {exc}')
                            failed_indices.add(index)
                except concurrent.futures.TimeoutError:
                    logging.error("The entire batch get_concurrent_response has timed out")
                finally:
                    executor.shutdown(wait=True)

            if len(results) == len(prompts):
                break
            else:
                logging.warning('Some requests failed, retrying...')
                retries += 1

        if len(results) != len(prompts):
            logging.error('Max retries reached, some requests failed.')
            raise Exception('get_concurrent_response failed, try to check the network or remote server.')

        return [results.get(i) for i in range(len(prompts))]