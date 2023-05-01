

class SpatResProcesser():
    """ Can be called to aggregate or interpolate files on the spatial axis.
    """
    pass

class TempResProcesser():
    """ Can be called to aggregate or interpolate files on the temporal axis.
    """
    pass

class EmissionProcesser():
    """ Can be called to summarize emission over sectors.
    """
    pass

class UnitsProcesser():
    """ Can be called to transform units.
    """
    # TODO add Charlie's code here if possible
    # don't forget the calendar inconsistencies
    pass

class XmipProcesser():
    """ Can be called to apply xmip preprocessing.
    """

# TODO find out where to put the co2 preprocessing
# def Cumsum co2 --> adapt input4mips
# def baseline --> substract, depending on experiment and cmip6 model
# TODO add important stuff in ghgs.json (which ghg are long living?)
class CO2Preprocesser():
    """ See: https://github.com/duncanwp/ClimateBench/blob/main/prep_input_data.ipynb
    """
