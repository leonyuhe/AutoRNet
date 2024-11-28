import logging
import re
from openai import OpenAI
import json
from typing import List , Dict
import concurrent.futures
from AutoRNet import Util
class Connector:
    def __init__(self, max_workers=4, **kwargs):
        try:
            self.model = kwargs['MODEL']
            self.client = OpenAI(api_key=kwargs['CONNECT_API_KEY'], base_url=kwargs['CONNECT_BASE_URL'])
            self.max_workers = max_workers
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def get_response(self, prompt_content):
        json_content = ''
        try:
            response = self.client.chat.completions.create(
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

    def get_concurrent_response(self, prompts: List, max_retries: int = 3):
        retries = 0
        results: Dict[int, str] = {}  # 使用字典来存储成功的结果，键是索引，值是结果
        failed_indices = set(range(len(prompts)))  # 初始化所有索引为失败的

        while retries < max_retries and failed_indices:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                try:
                    future_to_index = {executor.submit(self.get_response, prompts[i]): i for i in failed_indices}
                    failed_indices.clear()  # 清空失败索引，以便重新记录新的失败请求
                    for future in concurrent.futures.as_completed(future_to_index, timeout=Util.http_concurrent_timeout):
                        index = future_to_index[future]
                        try:
                            result = future.result(timeout=Util.concurrent_timeout)
                            results[index] = result  # 将结果存储在字典中
                        except Exception as exc:
                            logging.error('get_concurrent_response at index %d generated an exception: %s' % (index, future))
                            failed_indices.add(index)  # 记录失败的索引
                except concurrent.futures.TimeoutError:
                    logging.error("The entire batch get_concurrent_response has timed out")
                finally:
                    executor.shutdown(wait=True)

            # 检查是否所有请求都成功
            if len(results) == len(prompts):
                break
            else:
                logging.warning('Some requests failed, retrying...')
                retries += 1

        # 如果最终结果数量仍然不匹配，记录错误日志
        if len(results) != len(prompts):
            logging.error('Max retries reached, some requests failed.')
            raise Exception('get_concurrent_response failed, try to check the network or remote server.')

        # 按照原始顺序返回结果列表
        return [results.get(i) for i in range(len(prompts))]