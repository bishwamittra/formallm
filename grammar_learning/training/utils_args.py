import argparse
import json


def get_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--checkpoint_path_overwrite", type=str, default=None, help="Checkpoint path overwrite")
    parser.add_argument("--use_untrained_model", action='store_true', help="Use untrained model")
    parser.add_argument("--grammar_name", type=str, default="anbn", help="Grammar name")
    parser.add_argument("--data_comment", type=str, default=None, help="Data comment")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of sequences")
    parser.add_argument("--learning_rate", type=float, default=0.00005, help="Learning rate")
    parser.add_argument("--comment", type=str, default="", help="Comment")
    parser.add_argument("--generate_text", action='store_true', help="Store result")
    parser.add_argument("--considered_training_samples", type=int, default=None, help="Considered training samples")
    parser.add_argument("--skip_training_samples", type=int, default=0, help="Skip training samples")
    parser.add_argument("--considered_eval_samples", type=int, default=128, help="Considered training samples")
    parser.add_argument("--considered_incontext_examples", type=int, default=0, help="Considered incontext samples")
    parser.add_argument("--considered_incontext_repetitions", type=int, default=1, help="How many times to repeat in-context experiments")
    parser.add_argument("--incontext_data_source", type=str, default=None, help="Source pickle file and dataset name separated by colon")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=4, help="Logging steps")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["epoch", "steps"], help="Evaluation strategy")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps for training")
    parser.add_argument("--data_seed", type=int, default=5, help="Random seed for data")
    parser.add_argument("--run_seed", type=int, default=None, help="Random seed for experiments")
    parser.add_argument("--save_checkpoint", action='store_true', help="Save checkpoint")
    parser.add_argument("--save_final_checkpoint", action='store_true', help="Save final checkpoint")
    parser.add_argument("--save_best_model", action='store_true', help="Save the best checkpoint")
    parser.add_argument("--incontext_input", action='store_true', help="Process input as incontext input")
    parser.add_argument("--use_deepspeed", action='store_true', help="Use deepspeed")
    parser.add_argument("--max_new_tokens", default=1, type=int, help="Max new tokens to generate in text generation mode")
    parser.add_argument("--compute_msp", action='store_true', help="Compute minimum sufficient prefix")
    parser.add_argument("--incontext_separator", type=str, default="semicolon", help="Separator in in-context learning experiemnts")
    parser.add_argument("--run_in_docker", action='store_true', help="Run in docker")
    parser.add_argument("--lr_scheduler", type=str, default="linear", help="Learning schedule", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")

    # dataset selection
    parser.add_argument("--exclude_test_data", action='store_true', help="Exclude test data")
    parser.add_argument("--include_edit_distance_eval", action='store_true', help="Include edit distance eval datasets")
    parser.add_argument("--include_edit_distance_1_eval", action='store_true', help="Include edit distance eval datasets")
    parser.add_argument("--include_grammar_edit_eval", action='store_true', help="Include grammar_edit eval datasets")
    parser.add_argument("--include_incorrect_random_eval", action='store_true', help="Include incorrect random eval datasets")
    parser.add_argument("--combine_edit_distance", action='store_true', help="Combine edit distance datasets")
    parser.add_argument("--include_grammar_rule_data", action='store_true', help="Include grammar rule data (both correct and incorrect)")
    

    # counterfactual memorization
    parser.add_argument("--counterfactual_memorization", action='store_true', help="Counterfactual memorization")
    parser.add_argument("--counterfactual_string_index", type=int, default=0, help="Counterfactual string index")
    parser.add_argument("--mem_no_batch", action='store_true', help="Whether to put all training strings in one batch or not")

    parser.add_argument("--use_under_trained_tokens", action='store_true', help="Use untrained tokens")
    parser.add_argument("--icl_batch_size", type=int, default=8, help="Batch size for ICL")

    # local prefix
    parser.add_argument("--global_prefix_config", type=str, default='no_global_prefix', help="Configuration of global prefix")

    # memorization-aware training
    parser.add_argument("--memorization_algo", type=str, default='no_intervention')  # deduplication, remove_after_memorized, remove_after_memorized_and_add_when_forgot

    # nlp dataset & instruction
    parser.add_argument("--nlp_dataset", action='store_true', help="NLP dataset")
    parser.add_argument("--add_instruction", action='store_true', help="Add instruction")
    parser.add_argument("--instruction_index", default=0, type=int, help="Instruction index")
    parser.add_argument("--instruction", default=None, type=str, help="Instruction")

    # multilingual training
    parser.add_argument("--multilingual", action='store_true', help="Multilingual training")
    parser.add_argument("--multilingual_grammar_name", type=str, default=None, help="Multilingual grammar name (not used now)")
    parser.add_argument("--multilingual_ratio", default=1.0, type=float, help="Multilingual ratio between two languages")

    # use model tokens (for extended training)
    parser.add_argument("--map_to_model_specific_tokens", action='store_true', help="Map to model specific tokens")

    parser.add_argument("--adaptive_training", action='store_true', help="Whether to use adaptive training")
    parser.add_argument("--token_map_reverse", default=None, help="Token map reverse")

    parser.add_argument("--discriminative_individual", action='store_true', help="Test data contains grammatically incorrect sequences -- used for evaluating language learning locally")

    # reference model (to compare with baseline)
    parser.add_argument("--reference_model_results", type=str, default=None, help="Reference model")

    # architectural overrides (applied to config before from_config; implies --use_untrained_model)
    parser.add_argument("--arch_config_overrides", type=str, default=None, help="JSON dict of config fields to override, e.g. '{\"num_hidden_layers\": 2, \"rotary_pct\": 0.5}'")

    # process
    args, _ = parser.parse_known_args()
    assert args.max_steps != -1 or args.num_train_epochs != -1, "Either max_steps or num_train_epochs should be specified"

    if args.arch_config_overrides is not None:
        try:
            json.loads(args.arch_config_overrides)
        except json.JSONDecodeError as e:
            parser.error(f"--arch_config_overrides is not valid JSON: {e}")
        args.use_untrained_model = True

    if args.incontext_input:
        args.use_deepspeed = False

    assert args.multilingual_grammar_name is None
    assert args.token_map_reverse is None
    if args.adaptive_training:
        args.evaluation_strategy = "steps"
        args.num_train_epochs = 1

    return args


separator_dict = {
    "space": " ",
    "semicolon": ";",
    "comma": ",",
    "colon": ":",
    "period": ".",
    "double_newline": "\n\n"
}