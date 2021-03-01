import json
import os

def get_config_from_json(json_file):
  """
  Get the config from a JSON file
  Parameters
  ----------
  json_file: path
    File path to JSON file.

  Returns
  -------
  config: dict
    Dictionary object of configs in the JSON file.
  """
  # parse the configurations from the config JSON file provided
  with open(json_file, 'r') as config_file:
    config_dict = json.load(config_file)
  return config_dict

def process_config(json_file):
  """
  Add more configs to the JSON config file.

  Parameters
  ----------
  json_file: path
    File path to JSON file.

  Returns
  -------
  config: dict
    Dictionary object of configs in the JSON file after being processed.
  """
  config = get_config_from_json(json_file)
  # config.summary_dir = os.path.join("../experiments", config.exp_name, "summary/")
  # config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint/")
  return config