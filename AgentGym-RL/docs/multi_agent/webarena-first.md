# WebArena-First Validation Order

1. `python3 scripts/multi_agent/preflight.py --task webarena --mode eval --gpus 1`
2. `bash examples/eval/MultiAgent/webarena_eval.sh data.batch_size=1`
3. `python3 scripts/multi_agent/preflight.py --task webarena --mode train --gpus 8`
4. `bash examples/train/MultiAgent/webarena_train.sh trainer.total_epochs=1 data.train_batch_size=8`
5. After WebArena smoke passes, repeat evaluation on `sciworld`, `searchqa`, `babyai`, and `textcraft`.
