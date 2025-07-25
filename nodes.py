
import os
import random
import re
import datetime
import folder_paths
import comfy.utils, comfy.sample, comfy.samplers, comfy.controlnet, comfy.model_base, comfy.model_management, comfy.sampler_helpers, comfy.supported_models

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

lazy_options = {"lazy": True}

###
### Checkpoint の読み込み設定と、kSamplerの設定をひとまとめにする
###
class CheckPointSettingsPack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints") + ['None'],),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_skip": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT",{"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",)
            }
        }
        
    RETURN_TYPES = ("CP_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "packSettings"
    CATEGORY = "CheckPointSettings"
    
    def packSettings(self, ckpt_name, vae_name, clip_skip
                     , steps, cfg, sampler_name, scheduler, denoise
                     , optional_lora_stack=None):
        
        if ckpt_name == 'None':
            return (None,)
        
        checkpoint_settings =  {
            "optional_lora_stack": optional_lora_stack,
            "ckpt_name": ckpt_name,
            "vae_name": vae_name,
            "clip_skip": clip_skip,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
        }
        return (checkpoint_settings,)

###
### 複数の設定をまとめてリスト化する
###
class CheckPointSettingsTie:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
            },
            "optional": {
            }
        }
        inputs["optional"]["base_settings_list"] = ("CP_SETTINGS_LIST",)
        for i in range(10):
            inputs["optional"]["settings%d" % i] = ("CP_SETTINGS",)
        return inputs
        
    RETURN_TYPES = ("CP_SETTINGS_LIST","INT",)
    RETURN_NAMES = ("settings_list","size",)
    FUNCTION = "tie"
    CATEGORY = "CheckPointSettings"
       
    def tie(self, **kwargs):
        output_list = []
        for k, v in kwargs.items():
            if k == "base_settings_list" and v is not None:
                output_list.extend(v)
            if k.startswith("settings") and v is not None:
                if v['ckpt_name'] != 'None':
                    output_list.append(v)
        
        return (output_list, len(output_list),)

###
### 複数の設定を合体させる
###
class CheckPointSettingsListMerger:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
            },
            "optional": {
            }
        }
        for i in range(5):
            inputs["optional"]["settings_list%d" % i] = ("CP_SETTINGS_LIST",)
        return inputs
        
    RETURN_TYPES = ("CP_SETTINGS_LIST","INT",)
    RETURN_NAMES = ("settings_list","size",)
    FUNCTION = "tie"
    CATEGORY = "CheckPointSettings"

    def tie(self, **kwargs):
        output_list = []
        for k, v in kwargs.items():
            if k.startswith("settings_list") and v is not None:
                output_list.extend(v)
        
        return (output_list, len(output_list),)

###
### 設定リストからランダムに1つを選択する.
###
class CheckPointSettingsRandomSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings_list": ("CP_SETTINGS_LIST", ),
                "seed": ("INT",)
            },
            "optional": {
            }
        }
    
    RETURN_TYPES = ("CP_SETTINGS","INT",)
    RETURN_NAMES = ("settings","index",)
    FUNCTION = "index_switch"
    CATEGORY = "CheckPointSettings"
            
    def index_switch(self, settings_list, seed):
        instance = random.SystemRandom(seed)
        index = int(instance.randrange(len(settings_list)))
        value = settings_list[index]
        return (value, index,)
    
###
### まとめた情報を取り出す
###    
class CheckPointSettingsUnpack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings": ("CP_SETTINGS",),
            }
        }
        
    RETURN_TYPES = ("CP_SETTINGS", "LORA_STACK"
                    , AlwaysEqualProxy("*"), AlwaysEqualProxy("*"), "INT"
                    , "INT", "FLOAT", AlwaysEqualProxy("*"), AlwaysEqualProxy("*"), "FLOAT"
                    , )
    RETURN_NAMES = ("settings", "optional_lora_stack"
                    , "ckpt_name", "vae_name", "clip_skip"
                    , "steps", "cfg", "sampler_name", "scheduler", "denoise"
                    ,)
    FUNCTION = "unpackSettings"
    CATEGORY = "CheckPointSettings"
    
    def unpackSettings(self, settings):
        return (settings, settings["optional_lora_stack"]
                , settings["ckpt_name"], settings["vae_name"], settings["clip_skip"]
                , settings["steps"], settings["cfg"], settings["sampler_name"], settings["scheduler"],settings["denoise"])

###
### 設定情報からファイル名を作成
###
class CheckPointSettingsToFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings": ("CP_SETTINGS",),
                "format": ("STRING", {"default": "{ckpt_name}_{date}"})
            }
        }
        
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("filename", )
    FUNCTION = "makeFilename"
    CATEGORY = "CheckPointSettings"
    
    def makeFilename(self, settings, format):
        fmtParam = settings.copy()
        ckpt_name = fmtParam['ckpt_name']
        ckpt_name = os.path.basename(os.path.splitext(ckpt_name)[0])
        fmtParam['ckpt_name'] = ckpt_name
        vae_name = fmtParam['vae_name']
        if vae_name != "Baked VAE":
            vae_name = os.path.basename(os.path.splitext(vae_name)[0])
        fmtParam['vae_name'] = vae_name
        currenttime = datetime.datetime.now()
        fmtParam['datetime'] = currenttime
        fmtParam['date'] = currenttime.date()
        fmtParam['time'] = currenttime.time()
        result = re.sub(r'[\\|/|:|?|.|"|<|>|\|]', '-', format.format(**fmtParam))
        return (result,)
        
###
### 設定された値からランダムで数値を選択する. LORAの効きを調整する等の用途で利用
###
class RandomChoiceNumber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_list": ("STRING", {"default": "0.0|0.125|0.25" }),
                "seed": ("INT",)
            },
            "optional": {
            }
        }
    
    RETURN_TYPES = ("FLOAT","INT","INT",)
    RETURN_NAMES = ("selected_value","rounded_int", "index",)
    FUNCTION = "index_switch"
    CATEGORY = "CheckPointSettings/Utility"
            
    def index_switch(self, float_list, seed):
        splited = float_list.split("|")
        instance = random.SystemRandom(seed)
        index = int(instance.randrange(len(splited)))
        value = splited[index]
        fval = float(value)
        ival = int(round(fval))
        return (fval, ival, index,)    


NODE_CLASS_MAPPINGS = {
    "CheckPointSettingsPack": CheckPointSettingsPack,
    "CheckPointSettingsUnpack": CheckPointSettingsUnpack,
    "CheckPointSettingsTie": CheckPointSettingsTie,
    "CheckPointSettingsListMerger": CheckPointSettingsListMerger,
    "CheckPointSettingsRandomSelector": CheckPointSettingsRandomSelector,
    "CheckPointSettingsToFilename": CheckPointSettingsToFilename,
    "RandomChoiceNumber": RandomChoiceNumber,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckPointSettingsPack": "CheckPointSettingsPack",
    "CheckPointSettingsUnpack": "CheckPointSettingsUnpack",
    "CheckPointSettingsTie": "CheckPointSettingsTie",
    "CheckPointSettingsListMerger": "CheckPointSettingsListMerger",
    "CheckPointSettingsRandomSelector": "CheckPointSettingsRandomSelector",
    "CheckPointSettingsToFilename": "CheckPointSettingsToFilename",
    "RandomChoiceNumber": "RandomChoiceNumber",
}