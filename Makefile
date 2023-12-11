# Run all commands in one shell
.ONESHELL:

# Default target
.DEFAULT_GOAL := help

.PHONY : help
## help: run 'make help" at commandline
help : Makefile
	@sed -n 's/^##//p' $<

.PHONY: list
## list: list all targets in the current make file
list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Generic Variables
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d")

# Notes: Since accelerate is not working, I recommend using the following values
# | Model | Batch | CPU | Memory |
# | 1 | 688 | 3 | 96 | (ignore this model going forward)
# | 2 | 240 | 3 | 96 |

# Training parameters
model_name = simple_unet segformer
loss_type = dice
num_epochs = 100
augment = 0
lr = 5e-5
# debug = 0
batch_size = 512
nr_of_classes = 51 107


## ddpm-train: train a model from scratch
tl-train:
	for model in $(model_name); do \
		for loss in $(loss_type); do \
			logdir=20231211-M$$model\L$$loss\C$(nr_of_classes)\B$(batch_size)\A$(augment)
			sbatch --job-name=$$logdir submit.sh python -u scripts/commands/main.py train \
				--model_name $$model \
				--nr_of_classes $(nr_of_classes) \
				--logdir $$logdir \
				--num_epochs $(num_epochs) \
				--batch_size $(batch_size) \
				--augment $(augment) \
				--lr $(lr); \
		done; \
	done;


## ddpm-resume: resume training
tl-resume:
	for model in $(model_name); do \
		for loss in $(loss_type); do \
			logdir=test-M$$model\L$$loss\A$(augment)
			sbatch --job-name=$$logdir --open-mode=append submit.sh python -u scripts/commands/main.py resume-train \
				/space/calico/1/users/Harsha/ddpm-labels/logs/$$logdir; \
		done; \
	done;


## tl-test: test changes to code using fashion-mnist data
tl-test:
	python -u scripts/main.py train \
		--model_name segformer \
		--logdir mnist \
		--num_epochs 10 \
		--debug 1\
		;


## model-summary: print model summary
model-summary:
	python TissueLabeling/models/segformer.py
	python TissueLabeling/models/unet.py