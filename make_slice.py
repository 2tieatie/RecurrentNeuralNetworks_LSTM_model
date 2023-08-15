def make_slice_back(df, target_date: str, n: int):
    closest_date = df.index[df.index <= target_date][-1]
    closest_date_index = df.index.get_loc(closest_date)
    start_index = max(0, closest_date_index - n + 1)
    sliced_df = df.iloc[start_index:closest_date_index + 1]
    return sliced_df


def make_slice(df, target_date: str, n: int):
    closest_date = df.index[df.index <= target_date][-1]
    closest_date_index = df.index.get_loc(closest_date)
    end_index = min(df.shape[0], closest_date_index + n)
    sliced_df = df.iloc[closest_date_index:end_index]
    return sliced_df