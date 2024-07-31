from enum import Enum
import config
import os
import openai
#adding
from openai import AzureOpenAI

try:
    # from openai import OpenAI #imports openai class
    from openai import AzureOpenAI
    o = AzureOpenAI(api_key='Your-OpenAI-API key', api_version='2024-06-01',azure_endpoint='Your-OpenAI-API endpoint') #creates instance of the class with the dummy api key
    # DEFAULT_OPENAI_API_BASE = o.base_url # base URL of the OpenAI API
    DEFAULT_OPENAI_API_BASE = os.getenv('AZURE_OPENAI_ENDPOINT')  # base URL of the Azure OpenAI API

    del o
except ImportError:
    AzureOpenAI = None
class response_type(Enum):
    MESSAGE = 1
    TOOL_CALL = 2

def prompt_to_chat(prompt, system=None):
    '''
    Convert a prompt string to a chat-style message list

    Args:
        prompt (str): Prompt to convert

    Returns:
        list: single item with just the given prompt as user message for chat history
        e.g. messages=[{"role": "user", "content": "Hello world"}])
    '''
    messages = [] if system is None else [{'role': 'system', 'content': system}]
    messages.append({'role': 'user', 'content': prompt})
    return messages

class llm_response(config.attr_dict):
    '''
    Uniform interface for LLM responses from Open
    '''
    @staticmethod #does not require an instance of the class to be called and does not modify class or instance state.
    def from_openai_chat(response):
        '''
        Convert an OpenAI API ChatCompletion object to an llm_response object
        '''
        # print(f'from_openai_chat: {response =}')
        resp = llm_response(response)
        resp['response_type'] = response_type.MESSAGE  # Default assumption #from enum class :message=1
        if 'usage' in resp:
            resp['usage'] = llm_response(resp['usage'])
            resp['prompt_tokens'] = resp.usage.prompt_tokens
            resp['generated_tokens'] = resp.usage.completion_tokens

        elif 'timings' in resp:
            resp['timings'] = llm_response(resp['timings'])
            resp['prompt_tokens'] = resp.timings.prompt_n
            resp['generated_tokens'] = resp.timings.predicted_n
            resp['prompt_tps'] = resp.timings.prompt_per_second
            resp['generated_tps'] = resp.timings.predicted_per_second
        if resp.get('choices', []):
            resp['choices'] = [llm_response(c) for c in resp['choices']]
            for c in resp['choices']:
                if 'message' in c:
                    c['message'] = llm_response(c['message'])
            rc1 = resp['choices'][0]
            # No response message content if a tool call is invoked
            if rc1.get('message', {}).get('tool_calls'):
                resp['response_type'] = response_type.TOOL_CALL #from enum class for responsetype, value is 2
                for tc in rc1['message']['tool_calls']:
                    tc['function']['arguments_obj'] = json.loads(tc['function']['arguments'])#convert this JSON string into a Python dictionar
            else:
                resp['first_choice_text'] = rc1['text'] if 'text' in rc1 else rc1['message'].get('content', '')
            # print(f'from_openai_chat: {rc1 =}')
        else:
            resp['first_choice_text'] = resp['content']
        return resp

    from_llamacpp = from_openai_chat
    # llama_cpp regular completion endpoint response keys: 'content', 'generation_settings', 'model', 'prompt', 'slot_id', 'stop', 'stopped_eos', 'stopped_limit', 'stopped_word', 'stopping_word', 'timings', 'tokens_cached', 'tokens_evaluated', 'tokens_predicted', 'truncated'  # noqa
class llm_wrapper:
    '''
    Base-level wrapper for LLMs
    '''
    def __init__(self, model=None, **kwargs):
        '''
        Args:
            model (str): Name of the model being wrapped

            kwargs (dict, optional): Extra parameters for the API, the model, etc.
        '''
        self.model = model
        self.parameters = kwargs


# Based on class BaseClient defn in https://github.com/openai/openai-python/blob/main/src/openai/_base_client.py
# plus class OpenAI in https://github.com/openai/openai-python/blob/main/src/openai/_client.py
OPENAI_KEY_ATTRIBS = ['api_key', 'base_url', 'organization', 'timeout', 'max_retries']
class openai_api(llm_wrapper):
    '''
    Wrapper for LLM hosted via OpenAI-compatible API (including OpenAI proper).
    Designed for models that provide simple completions from prompt.
    For chat-style models (including OpenAI's gpt-3.5-turbo & gpt-4), use openai_chat_api
    '''
    def __init__(self, model=None, base_url=None, api_key=None, **kwargs):
        '''
        If using OpenAI proper, you can pass in an API key, otherwise environment variable
        OPENAI_API_KEY will be checked

        Args:
            model (str, optional): Name of the model being wrapped. Useful for using
            OpenAI proper, or any endpoint that allows you to select a model

            base_url (str, optional): Base URL of the API endpoint

            api_key (str, optional): OpenAI API key to use for authentication

            kwargs (dict, optional): Extra parameters for the API or for the model host
        '''
        if AzureOpenAI is None:
            raise ImportError('openai module not available; Perhaps try: `pip install openai`')
        if api_key is None:
            api_key = os.getenv('AZURE_OPENAI_API_KEY', 'dummy')
            # api_key = os.getenv('OPENAI_API_KEY', config.OPENAI_KEY_DUMMY)

        self.api_key = api_key
        self.parameters = config.attr_dict(kwargs)
        self.base_url = base_url or os.getenv('AZURE_OPENAI_ENDPOINT')
        # if base_url:
        #     # If the user includes the API version in the base, don't add it again
        #     scheme, authority, path, query, fragment = iri.split_uri_ref(base_url)
        #     path = path or kwargs.get('api_version', '/v1')
        #     self.base_url = iri.unsplit_uri_ref((scheme, authority, path, query, fragment))
        # else:
        #     self.base_url = DEFAULT_OPENAI_API_BASE
        self.original_model = model or None
        self.model = model

    def call(self, prompt, api_func=None, **kwargs):
        '''
        Non-asynchronous entry to Invoke the LLM with a completion request

        Args:
            prompt (str): Prompt to send to the LLM

            kwargs (dict, optional): Extra parameters to pass to the model via API.
                See Completions.create in OpenAI API, but in short, these:
                best_of, echo, frequency_penalty, logit_bias, logprobs, max_tokens, n
                presence_penalty, seed, stop, stream, suffix, temperature, top_p, user

        Returns:
            dict: JSON response from the LLM
        '''
        print(f'prompt: {prompt}')
        # oai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        oai_client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),api_key=os.getenv("AZURE_OPENAI_API_KEY"), api_version="2024-06-01")
        # api_func = api_func or oai_client.completions.create
        api_func = oai_client.chat.completions.create(
        model="US-CDW-GPT4o-MODEL-cd",  # model = "deployment_name".
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
      ]
)

        #print(api_func.choices[0].message.content)

        merged_kwargs = {**self.parameters, **kwargs}
        if 'model' in merged_kwargs:
            del merged_kwargs['model']

        result = api_func(model=self.model, prompt=prompt, **merged_kwargs)
        result = llm_response.from_openai_chat(result)
        if result.model == 'HOSTED_MODEL':
            result.model = self.hosted_model()
        return result

    async def __call__(self, prompt, api_func=None, **kwargs):
        '''
        Invoke the LLM with a completion request

        Args:
            prompt (str): Prompt to send to the LLM

            kwargs (dict, optional): Extra parameters to pass to the model via API.
                See Completions.create in OpenAI API, but in short, these:
                best_of, echo, frequency_penalty, logit_bias, logprobs, max_tokens, n
                presence_penalty, seed, stop, stream, suffix, temperature, top_p, userq
        Returns:
            dict: JSON response from the LLM
        '''
        return self.call(prompt, api_func, **kwargs)

    def wrap_for_multiproc(self, prompt, **kwargs):
        '''
        Wrap the LLM invocation in an asyncio task

        Returns:
            asyncio.Task: Task for the LLM invocation
        '''
        merged_kwargs = {**self.parameters, **kwargs}
        return asyncio.create_task(
            schedule_callable(self, prompt, **merged_kwargs))

    def hosted_model(self) -> str:
        '''
        Model introspection: Query the API to find what model is being run for LLM calls
        '''
        if self.original_model:
            return self.original_model
        return self.available_models()[0]

    def available_models(self):
        '''
        Query the API to find what model is being run for LLM calls

        '''
        try:
            import httpx  # noqa
        except ImportError:
            raise RuntimeError('Needs httpx installed. Try pip install httpx')

        resp = httpx.get(f'{self.base_url}/models').json()
        if 'data' not in resp:
            raise RuntimeError(f'Unexpected response from {self.base_url}/models:\n{repr(resp)}')
        return [ i['id'] for i in resp['data'] ]

class openai_chat_api(openai_api): #subclass (openai_chat_api) inherits from parent class (openai_api).
    '''
    Wrapper for a chat-style LLM hosted via OpenAI-compatible API (including OpenAI proper).
    Supports local chat-style models as well as OpenAI's gpt-3.5-turbo & gpt-4

    You need to set an OpenAI API key in your environment, or pass it in, for this next example

    '''
    def call(self, messages, api_func=None, **kwargs): #**unpacks the dictionary
        '''
        Non-asynchronous entry to Invoke the LLM with a completion request

        Args:
            messages (list): Series of messages representing the chat history
            e.f. messages=[{"role": "user", "content": "Hello world"}])

            kwargs (dict, optional): Extra parameters to pass to the model via API

        Returns:
            dict: JSON response from the LLM
        '''
        #print(f'messages: {messages}')
        # oai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        oai_client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),api_key=os.getenv("AZURE_OPENAI_API_KEY"),api_version="2024-06-01")
        api_func = oai_client.chat.completions.create(
        model="US-CDW-GPT4o-MODEL-cd",  # model = "deployment_name".
        messages= messages
 
)
        #print(api_func.choices[0].message.content)
        merged_kwargs = {**self.parameters, **kwargs} # Combines two dictionaries: self.parameters and kwargs. Any overlapping keys will have their values from kwargs. #self.parameters:dictionary of default parameters of the class instance
        if 'model' in merged_kwargs:
            del merged_kwargs['model'] #If the key 'model' exists, it is removed from the dictionary.

        result = api_func
        result = llm_response.from_openai_chat(result)
        if result.model == 'HOSTED_MODEL':
            result.model = self.hosted_model()
        return result

    async def __call__(self, prompt, api_func=None, **kwargs):
        '''
        Invoke the LLM with a completion request

        Args:
            prompt (str): Prompt to send to the LLM

            kwargs (dict, optional): Extra parameters to pass to the model via API.
                See Completions.create in OpenAI API, but in short, these:
                best_of, echo, frequency_penalty, logit_bias, logprobs, max_tokens, n
                presence_penalty, seed, stop, stream, suffix, temperature, top_p, user

        Returns:
            dict: JSON response from the LLM
        '''
        return self.call(prompt, api_func, **kwargs)

