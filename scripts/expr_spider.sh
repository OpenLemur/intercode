# python -m experiments.eval_n_turn \
#     --data_path ./data/sql/spider/ic_spider_dev.json \
#     --dialogue_limit 5 \
#     --env sql \
#     --image_name docker-env-sql \
#     --log_dir logs/experiments \
#     --max_turns 10 \
#     --policy chat \
#     --template game_sql \
#     --model gpt-4 \
#     --handicap  \
#     --verbose

# python -m experiments.eval_n_turn \
#     --data_path ./data/sql/spider/ic_spider_dev.json \
#     --dialogue_limit 5 \
#     --env sql \
#     --image_name docker-env-sql-spider \
#     --log_dir logs/experiments \
#     --max_turns 10 \
#     --policy hf_chat \
#     --template game_sql \
#     --model llama-2-70b-chat \
#     --handicap  \
#     --verbose


# python -m experiments.eval_n_turn \
#     --data_path ./data/sql/spider/ic_spider_dev.json \
#     --dialogue_limit 5 \
#     --env sql \
#     --image_name docker-env-sql-spider \
#     --log_dir logs/experiments \
#     --max_turns 10 \
#     --policy hf_chat \
#     --template game_sql \
#     --model lemur-70b-chat-v1 \
#     --handicap  \
#     --verbose

# python -m experiments.eval_n_turn \
#     --data_path ./data/sql/spider/ic_spider_dev.json \
#     --dialogue_limit 5 \
#     --env sql \
#     --image_name docker-env-sql-spider \
#     --log_dir logs/experiments \
#     --max_turns 10 \
#     --policy hf_chat \
#     --template game_sql \
#     --model codellama-34b-instruct \
#     --handicap  \
#     --verbose
