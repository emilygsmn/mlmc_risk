"""Module providing data preprocessing helper functions."""

__all__ = ["preproc_portfolio"]

def _select_port_instr(port, instr_info):
    """Function selecting the relevant portfolio positions."""

    # Select the relevant (non-zero) positions from portfolio
    port = port[port.iloc[:,1] != 0.0]

    # Filter for relevant instrument data from meta dataframe
    instr_list = port.iloc[:, 0].tolist()
    instr_info = instr_info[instr_info.iloc[:, 0].isin(instr_list)]

    return port, instr_info

def _add_valuation_tag(instr_info):
    """Function categorizing the instruments by valuation method."""
    # BOND, BOND_FX, BOND_IF, BOND_IF_FX, BOND_CS, BOND_CS_FX, BOND_IF_CS, BOND_IF_CS_FX
    # DER_SWAP, DER_PUT
    # FX
    # ...

    # Define the conditions to assign the valuation tags by
    def _classify(name):
        if name.startswith("FX"):
            return "FX"
        if name.startswith("Other-EQ"):
            return "EQ"
        if "FI" in name:
            if name == "GOV-FI-AT-NA-NA-05":
                return "BOND"
            if name == "FI-GBP-RFR-NA-NA-NA-NA-01" or name == "GOV-FI-UK-NA-NA-05":
                return "BOND_FX"
        else:
            return "unknown"

    # Apply the classification to all instruments and save the tag in new column
    instr_info["val_tag"] = instr_info["fin_instr"].apply(_classify)

    return instr_info

def _get_calib_target(instr_info):
    """Function selecting the calibration targets from the instrument meta data."""
    return instr_info[["fin_instr", "calibration_target"]]

def preproc_portfolio(port, instr_info):
    """Function preprocessing the portfolio composition and instrument meta data."""

    # Select only the non-zero components from the portfolio
    #port, instr_info = _select_port_instr(port, instr_info)

    ### Preliminary filtering only for testing purposes:
    selected_positions = [
        "GOV-FI-AT-NA-NA-05",
        "GOV-FI-UK-NA-NA-05",
        #"FI-GBP-RFR-NA-NA-NA-NA-01",
        "Other-EQ-EUR-PUBL-EU-SX5T-NA-NA-NA"#,
        #"FX-GBP-NA-NA-NA-NA-NA-NA",
        #"FX-USD-NA-NA-NA-NA-NA-NA"
    ]
    port = port[port["fin_instr"].isin(selected_positions)]
    instr_info = instr_info[instr_info["fin_instr"].isin(selected_positions)]
    ###

    # Add a valuation tag
    instr_info = _add_valuation_tag(instr_info)

    # Get calibration target
    calib_target = _get_calib_target(instr_info)

    return port, instr_info, calib_target
