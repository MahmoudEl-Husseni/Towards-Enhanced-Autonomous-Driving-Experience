from config import *

def progress_bar(i, length=70, train_set_len=config.TRAIN_SZ, train_bs=config.TRAIN_BS):
  """
  Displays a progress bar indicating the completion of a training step.

  Args:
    i (int): The current training step.
    length (int, optional): The length of the progress bar in characters. Defaults to 70.

  Returns:
    A string representing the progress bar.
  """

  train_steps = (train_set_len / train_bs).__ceil__()

  progress = (i+1)/train_steps
  eq = '='
  progress_bar = f"{red}{'progress:'}{res} {(f'{(progress*100):.2f}'+' %').ljust(7)} [{f'{eq*int(i*length/train_steps)}>'.ljust(length)}]"
  return progress_bar