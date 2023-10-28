import re

from typing import Tuple

from .utils import ACTION_PARSER_MAP, PROMPT_MAP, CompletionGPT, ChatGPT, PalmChat, PalmCompletion, HFChat

class BasePolicy:
    def __init__(self):
        pass

    def forward(query, observation, available_actions):
        raise NotImplementedError
    
class HumanPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
    
    def forward(self, query, observation, available_actions):
        action = input('> ')
        return action
    
class CompletionGPTPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "text-davinci-003", response_limit: int = 500):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.response_limit = response_limit
        self.model = model
        self.handicap = ""
        self.dialogue_limit = dialogue_limit
    
    def reset(self):
        self.prompt = None
        self.dialogue = {}

    def add_to_dialogue(self, handicap: str):
        self.handicap = handicap

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        if self.prompt is None:
            # First Turn
            prompt = self.prompt = self.template.get_init_msg() + self.handicap + self.template.get_query_msg(query)
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            if isinstance(observation, str) and observation == "" or isinstance(observation, list) and len(observation) == 0:
                observation = "No output"
            self.dialogue['reward'] = reward
            self.dialogue['observations'] = observation
            # N-th Turn
            prompt = self.prompt + self.template.get_obs_msg(self.dialogue)

        # Retrieve Completion GPT
        actions = CompletionGPT(prompt, model=self.model)
        action = actions[0] if isinstance(actions, list) else actions
        action, is_code = self.action_parser(action)
        self.dialogue['actions'] = action
        return action, is_code

class ChatGPTPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "gpt-3.5-turbo", response_limit: int = 1000):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.dialogue_limit = dialogue_limit
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.model = model
        self.response_limit = response_limit
    
    def reset(self):
        self.dialogue = [{"role": "system", "content": self.template.get_init_msg()}]

    def add_to_dialogue(self, handicap: str):
        self.dialogue.append({"role": "system", "content": handicap})

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        # Append response to dialogue
        if self.dialogue[-1]["role"] == "system":
            # First Turn
            self.dialogue.append({"role": "user", "content": self.template.get_query_msg(query)})
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            # N-th Turn
            self.dialogue.append({"role": "user", "content": self.template.get_obs_msg(observation, reward)})
            # Only keep {self.dialogue_limit} most recent messages
            if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
                self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]

        # Retrieve Action from ChatGPT
        actions = ChatGPT(self.dialogue, model=self.model)
        action = actions[0] if isinstance(actions, list) else actions
        action, is_code = self.action_parser(action)
        self.dialogue.append({"role": "assistant", "content": action})
        return action, is_code
    
class PalmChatPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "models/chat-bison-001", response_limit: int = 1000):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.response_limit = response_limit
        self.model = model
        self.handicap = ""
    
    def reset(self):
        self.chatbot = PalmChat(self.model)
        self.dialogue = []

    def add_to_dialogue(self, handicap: str):
        self.handicap = handicap

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        if len(self.dialogue) == 0:
            # First Turn
            self.dialogue = [{"author": "0", "content": self.template.get_init_msg() + self.handicap + self.template.get_query_msg(query)}]
            self.chatbot.init_chat(init_message=self.dialogue)
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            # N-th Turn
            self.chatbot.reply(self.template.get_obs_msg(observation, reward))

        # Retrieve Action from PalmChat
        action = self.chatbot.get_response()
        action, is_code = self.action_parser(action)
        return action, is_code
    

class PalmCompletionPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "models/text-bison-001", response_limit: int = 1000):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.response_limit = response_limit
        self.model = model
        self.handicap = ""
    
    def reset(self):
        self.prompt = None
        self.dialogue = {}

    def add_to_dialogue(self, handicap: str):
        self.handicap = handicap

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        if self.prompt is None:
            # First Turn
            prompt = self.prompt = self.template.get_init_msg() + self.handicap + self.template.get_query_msg(query)
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            if isinstance(observation, str) and observation == "" or isinstance(observation, list) and len(observation) == 0:
                observation = "No output"
            self.dialogue['reward'] = reward
            self.dialogue['observations'] = observation
            # N-th Turn
            prompt = self.prompt + self.template.get_obs_msg(self.dialogue)

        # Retrieve Action from PalmChat
        actions = PalmCompletion(prompt, model=self.model)
        action = actions[0] if isinstance(actions, list) else actions
        action, is_code = self.action_parser(action)
        self.dialogue['actions'] = action
        return action, is_code

class HFChatPolicy(BasePolicy):
    def __init__(self, language: str, setting: str, template: str, dialogue_limit: int = None, model: str = "llama-2-70b-chat", response_limit: int = 1000):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.dialogue_limit = dialogue_limit
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.model = model
        self.response_limit = response_limit
        self.encode_message_func = None
        self.step_id = None
        if model in ["llama-2-70b-chat", "codellama-34b-instruct"]:
            def llama_encode_message(dialogue):
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                
                if dialogue[0]["role"] == "system":
                    dialogue = [
                            {
                                "role": dialogue[1]["role"],
                                "content": f"{B_SYS} {dialogue[0]['content']} {E_SYS} {dialogue[1]['content']}",
                            }
                        ] + dialogue[2:]
                
                chat = ""
                for d in dialogue:
                    if d["role"] == "user":
                        chat += f"{B_INST} {d['content']} {E_INST} "
                    elif d["role"] == "assistant":
                        chat += d['content'] + "\n"
                return chat
            
            self.encode_message_func = llama_encode_message
            self.stop_sequences = ["[INST]", "[/INST]", "Observation:", "Task:", "Output:"]
            self.post_process_func = lambda x: x.split("[/INST]")[0].strip()
        elif model in ["lemur-70b-chat-v1"]:
            def lemur_encode_message(dialogue):
                chat = ""
                for d in dialogue:
                    chat += f"<|im_start|>{d['role']}\n{d['content']}\n<|im_end|>\n"
                chat += "<|im_start|>assistant\n"
                return chat
            self.encode_message_func = lemur_encode_message
            self.stop_sequences = ["<|im_start|>", "<|im_end|>", "Observation:", "Task:", "Output:"]
            self.post_process_func = lambda x: x.split("<|im_end|>")[0].strip()
        else:
            raise NotImplementedError
    
    def reset(self):
        self.dialogue = []

    def add_to_dialogue(self, handicap: str):
        self.dialogue.append({"role": "system", "content": handicap})

    def prepare_dialogue(self, query, observation, reward):
        # Append response to dialogue
        if len(self.dialogue) == 0:
            self.dialogue += [{"role": "system", "content": f"You are a helpful assistant expert specializing in {self.language}."}]
            self.dialogue += [{"role": "user", "content": self.template.get_init_msg() + "\n" + self.template.get_query_msg(query)}]
        elif self.dialogue[-1]["role"] == "system":
            self.dialogue.append({"role": "user", "content": self.template.get_init_msg() + self.template.get_query_msg(query)})
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[:self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            # N-th Turn
            if self.step_id is not None:
                self.dialogue.append({"role": "user", "content": f"Observation {self.step_id}: " + self.template.get_obs_msg(observation, reward)})
            else:
                self.dialogue.append({"role": "user", "content": self.template.get_obs_msg(observation, reward)})
            # Only keep {self.dialogue_limit} most recent messages
            if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
                self.dialogue = self.dialogue[:2] + self.dialogue[-self.dialogue_limit:]

    def forward(self, query, observation, reward, available_actions) -> Tuple[str, bool]:
        self.prepare_dialogue(query, observation, reward)
        
        chat = self.encode_message_func(self.dialogue)
        if self.language == "CTF":
            action_prompt = "Action:"
            chat += action_prompt
        result = HFChat(chat, stop_sequences=self.stop_sequences)
        result = self.post_process_func(result)
        if self.language == "CTF":
            chat = chat[:-len(action_prompt)]
            result = action_prompt + " " + result
        action = result[0] if isinstance(result, list) else result
        action, is_code = self.action_parser(action)
        if self.language in ["SQL", "BASH", "PYTHON"] and is_code:
            self.dialogue.append({"role": "assistant", "content": f"```{self.language.lower()}\n{action}\n```"})
        elif self.language == "CTF":
            self.dialogue.append({"role": "assistant", "content": f"Action: {action}"})
        else:
            self.dialogue.append({"role": "assistant", "content": action})
        return action, is_code