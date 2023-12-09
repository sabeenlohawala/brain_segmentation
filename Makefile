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
model_idx = 1 2
epochs = 500
augment = 0
lr = 5e-5
im_size = (192, 224)
batch_size = 512


## ddpm-train: train a model from scratch
tl-train:
	for model in $(model_idx); do \
		for loss in $(loss_type); do \
			logdir=M$$model\T$(time_steps)$$schedule\L$$loss\G$(group_labels)J$(jei_flag)D$(downsampled)
			sbatch --job-name=$$logdir submit.sh python -u scripts/commands/main.py train \
				--model_idx $$model \
				--logdir $$logdir \
				--epochs $(epochs) \
				--batch_size $(batch_size) \
				--augment $(augment) \
				--lr $(lr) \
				--im_size '$(im_size)' \
				--downsample; \
		done; \
	done;


## ddpm-resume: resume training
tl-resume:
	for model in $(model_idx); do \
		for schedule in $(beta_schedule); do \
			for loss in $(loss_type); do \
				logdir=test-M$$model\T$(time_steps)$$schedule\L$$loss\G$(group_labels)J$(jei_flag)D1
				sbatch --job-name=$$logdir --open-mode=append submit.sh python -u scripts/commands/main.py resume-train \
					/space/calico/1/users/Harsha/ddpm-labels/logs/$$logdir; \
			done; \
		done; \
	done;


## tl-test: test changes to code using fashion-mnist data
tl-test:
	python -u scripts/main.py train \
		--model_idx 1 \
		--time_steps 30 \
		--beta_schedule linear \
		--logdir mnist \
		--epochs 10 \
		--im_size '(28, 28)' \
		--debug \
		;


## model-summary: print model summary
model-summary:
	python TissueLabeling/models/segformer.py
	python TissueLabeling/models/unet.py