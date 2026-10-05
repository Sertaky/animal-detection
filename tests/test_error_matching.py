from animal_detection.analysis import box_iou, match_image


def gt(label, box=(0, 0, 10, 10)):
    return {"label": label, "box": list(box)}


def pred(label, box=(0, 0, 10, 10), score=0.9):
    return {"label": label, "box": list(box), "score": score}


def test_iou_and_true_positive():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    result = match_image([gt(7)], [pred(7)])
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (1, 0, 0)
    assert result["predictions"][0]["iou"] == 1


def test_confusion_localization_and_background_are_distinct():
    ground_truth = [gt(3)]
    assert match_image(ground_truth, [pred(7)])["predictions"][0]["category"] == "class_confusion"
    assert match_image(ground_truth, [pred(3, (5, 0, 15, 10))])["predictions"][0]["category"] == "localization"
    assert match_image(ground_truth, [pred(3, (20, 20, 30, 30))])["predictions"][0]["category"] == "background"
    assert match_image(ground_truth, [pred(7)])["false_negative_gt_indices"] == [0]


def test_duplicate_higher_score_wins_even_if_input_order_is_reversed():
    result = match_image(
        [gt(3)],
        [pred(3, score=0.6), pred(3, score=0.9)],
    )
    assert [record["prediction_index"] for record in result["predictions"]] == [1, 0]
    assert [record["status"] for record in result["predictions"]] == ["true_positive", "false_positive"]
    assert result["predictions"][1]["category"] == "duplicate"


def test_empty_inputs_and_score_filter():
    assert match_image([], [])["predictions"] == []
    assert match_image([], [pred(3)])["predictions"][0]["category"] == "background"
    assert match_image([gt(3)], [])["false_negatives"] == 1
    result = match_image([gt(3)], [pred(3, score=0.49)])
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (0, 0, 1)


def test_multiple_objects_match_once_and_equal_scores_are_stable():
    result = match_image(
        [gt(3), gt(7, (20, 20, 30, 30))],
        [pred(3), pred(7, (20, 20, 30, 30)), pred(3)],
    )
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (2, 1, 0)
    assert [record["prediction_index"] for record in result["predictions"]] == [0, 1, 2]
    assert result["predictions"][2]["category"] == "duplicate"
