import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
import contextlib
import joblib



def prepare_panel(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    pre_treatment_period: Optional[Union[int, float]] = None,
) -> dict:
    """Prepare panel data for use in Synth.

    Args:
        df (pd.DataFrame): Dataframe
        unit_col (str): unit identifier
        time_col (str): time identifier
        outcome_col (str): outcome variable
        treatment_col (str): treatment variable
        pre_treatment_period (Optional[Union[int, float]], optional): pre-treatment period. Defaults to None.

    Returns:
        dict: Dictionary containing the following
            Y: Outcome matrix (N x T)
            W: Treatment matrix (N x T)
            treated_units: Array of treated unit indices
            T_pre: Number of pre-treatment periods
            unit_labels: Unique unit labels
            time_periods: Sorted array of time periods
    """
    # Basic data preparation
    Y, W, treated_units, T_pre = _prepare_panel_data(
        df, unit_col, time_col, outcome_col, treatment_col, pre_treatment_period
    )

    # Get additional information
    unit_labels = pd.Series(df[unit_col].unique()).sort_values()
    time_periods = pd.Series(df[time_col].unique()).sort_values()

    return {
        "Y": Y,
        "W": W,
        "treated_units": treated_units,
        "T_pre": T_pre,
        "unit_labels": unit_labels,
        "time_periods": time_periods,
    }


def _prepare_panel_data(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    pre_treatment_period: Optional[Union[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Convert long panel data to format required by Synth.

    Args:
        df: Panel DataFrame in long format
        unit_col: Name of unit ID column
        time_col: Name of time period column
        outcome_col: Name of outcome variable column
        treatment_col: Name of treatment indicator column
        pre_treatment_period: Optional time period denoting treatment start.
            If None, inferred as first period where treatment == 1

    Returns:
        Tuple of (Y, treated_units, T_pre) where:
            Y: Outcome matrix (N x T)
            treated_units: Array of treated unit indices
            T_pre: Number of pre-treatment periods

    Example:
        >>> df = pd.DataFrame({
        ...     'outcome': [1, 2, 3, 4, 5, 6],
        ...     'treatment': [0, 1, 0, 0, 0, 1],
        ...     'unit': [1, 1, 2, 2, 3, 3],
        ...     'time': [1, 2, 1, 2, 1, 2]
        ... })
        >>> Y, treated, T_pre = prepare_panel_data(
        ...     df, 'unit', 'time', 'outcome', 'treatment'
        ... )
    """
    # Check if panel is balanced
    units_per_time = df.groupby(time_col)[unit_col].count()
    if not (units_per_time == units_per_time.iloc[0]).all():
        raise ValueError("Panel must be balanced (same units in all time periods)")

    # Create outcome matrix
    Y = df.pivot(index=unit_col, columns=time_col, values=outcome_col).values

    # Get treatment matrix
    W = df.pivot(index=unit_col, columns=time_col, values=treatment_col).values

    # Identify treated units (any unit that receives treatment)
    treated_units = np.where(W.sum(axis=1) > 0)[0]

    # Determine pre-treatment period
    if pre_treatment_period is None:
        # Find first period where any unit is treated
        first_treated_period = df[df[treatment_col] == 1][time_col].min()
        # Convert to array index (assuming time periods are sorted)
        unique_times = sorted(df[time_col].unique())
        T_pre = unique_times.index(first_treated_period)
    else:
        # Find index of pre_treatment_period in sorted unique times
        unique_times = sorted(df[time_col].unique())
        T_pre = unique_times.index(pre_treatment_period)

    return Y, W, treated_units, T_pre

def convert_to_W(Y_treated: np.ndarray, Y_control: np.ndarray, T_pre: int) -> np.ndarray:
    """Convert observed outcomes to treatment matrix W.

    Args:
        Y_treated: Observed outcomes for treated units
        Y_control: Observed outcomes for control units
        T_pre: Number of pre-treatment periods

    Returns:
        W: Treatment matrix (N x T)
    """

    # if there are two dimensions first is number of treated units N_treated and second is number of time periods T otherwise N_treated = 1 and the dimention is the time periods
    if len(Y_treated.shape) == 1:
        Y_treated = Y_treated.reshape(1, -1)
    N_treated, T = Y_treated.shape
    N_control = Y_control.shape[0]

    # Initialize W as zeros
    W = np.zeros((N_treated + N_control, T))

    # Set treated units to 1 in post-treatment periods
    W[:N_treated, T_pre:] = 1

    # Union Y_treated and Y_control
    Y = np.vstack([Y_treated, Y_control])

    return Y, W, N_treated

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
