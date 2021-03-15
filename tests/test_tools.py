from project.utils.tools import select_data_with_max_length_sentence


def test_select_data_with_max_length_sentence():
    tokens_list = [[1], [1, 2], [1, 2, 3], [5]]
    colors = [[1], [2], [3], [4]]
    new_tokens, new_colors = select_data_with_max_length_sentence(2, tokens_list, colors)
    expected_tokens = [[1], [1, 2], [5]]
    expected_colors = [[1], [2], [4]]
    assert new_tokens == expected_tokens
    assert new_colors == expected_colors