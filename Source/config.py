# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0

'''
Configuration & globally-relevant values
'''

# OpenAI API requires the model be specified, but many compaitble APIs
# have a model predetermined by the host
HOST_DEFAULT_MODEL = HOST_DEFAULT = 'HOST-DEFAULT'
OPENAI_KEY_DUMMY = 'OPENAI_DUMMY'


class attr_dict(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

