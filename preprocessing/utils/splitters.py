from collections import Counter

import pandas as pd


def _build_target_counter(counter: Counter, train_size: float) -> tuple[Counter, Counter]:
    train_target = Counter()
    test_target = Counter()

    for label, total in counter.items():
        train_target[label] = int(round(total * train_size))
        train_target[label] = max(1, min(train_target[label], total - 1)) if total > 1 else total
        test_target[label] = total - train_target[label]

    return train_target, test_target


def stratified_category_instrument_split(
    df: pd.DataFrame,
    category_col: str,
    instruments_col: str,
    train_size: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Greedy split balancing:
    - single-label category distribution
    - multi-label instrument distribution
    """
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")

    work_df = df.copy().reset_index(drop=True)
    work_df = work_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    category_counter = Counter(work_df[category_col].tolist())
    instrument_counter = Counter()
    for insts in work_df[instruments_col].tolist():
        for inst in insts:
            instrument_counter[inst] += 1

    train_cat_target, test_cat_target = _build_target_counter(category_counter, train_size)
    train_inst_target, test_inst_target = _build_target_counter(instrument_counter, train_size)

    train_goal = int(round(len(work_df) * train_size))
    test_goal = len(work_df) - train_goal

    cat_freq = dict(category_counter)
    inst_freq = dict(instrument_counter)

    work_df = work_df.assign(
        __priority=work_df.apply(
            lambda row: (
                cat_freq.get(row[category_col], 0),
                sum(inst_freq.get(inst, 0) for inst in row[instruments_col]),
                -len(row[instruments_col]),
            ),
            axis=1,
        )
    ).sort_values("__priority").drop(columns=["__priority"])

    train_rows = []
    test_rows = []
    train_cat = Counter()
    test_cat = Counter()
    train_inst = Counter()
    test_inst = Counter()

    def score_train(category, instruments):
        cat_need = max(train_cat_target[category] - train_cat[category], 0)
        inst_need = sum(max(train_inst_target[i] - train_inst[i], 0) for i in instruments)
        return (3 * cat_need) + inst_need

    def score_test(category, instruments):
        cat_need = max(test_cat_target[category] - test_cat[category], 0)
        inst_need = sum(max(test_inst_target[i] - test_inst[i], 0) for i in instruments)
        return (3 * cat_need) + inst_need

    for _, row in work_df.iterrows():
        category = row[category_col]
        instruments = row[instruments_col]

        can_train = len(train_rows) < train_goal
        can_test = len(test_rows) < test_goal

        if not can_train:
            chosen = "test"
        elif not can_test:
            chosen = "train"
        else:
            train_score = score_train(category, instruments)
            test_score = score_test(category, instruments)

            if train_score == test_score:
                train_fill = (train_goal - len(train_rows)) / max(train_goal, 1)
                test_fill = (test_goal - len(test_rows)) / max(test_goal, 1)
                chosen = "train" if train_fill >= test_fill else "test"
            else:
                chosen = "train" if train_score > test_score else "test"

        if chosen == "train":
            train_rows.append(row)
            train_cat[category] += 1
            for inst in instruments:
                train_inst[inst] += 1
        else:
            test_rows.append(row)
            test_cat[category] += 1
            for inst in instruments:
                test_inst[inst] += 1

    train_df = pd.DataFrame(train_rows).reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).reset_index(drop=True)

    return train_df, test_df
