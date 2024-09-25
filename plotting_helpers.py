import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.size" : 12,
    #"pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": True,
    "lines.antialiased": True,
    "patch.antialiased": True,
    'axes.linewidth': 0.1
})


def set_size(width=455.24411, 
             fraction=1, 
             subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
            455.24411 seems to be the standard Latex width
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
    
    
# CHECK AGAIN IF THIS IS ALWAYS TRUE OR FORMAT DEPENDENT
LATEX_WIDTH = 455.24411 

blue = "#337AA1"  
blue2 = "#0CB5ED"
blue3 = "#1B2F57"
bluegrey = "#5A7585"
turq = "#24D6B1"  
turq2 = "#09736A"
red = "#A12D03"  
red2  = "#945A54"
red3 = "#E65849"
red4 = "#5C0000"
red5 = "#A03D4D" #A03D4D A1154D B84659
och = "#E0B25E"  
lime = "#AAE070"  
green = "#005C17"  
green2 = "#109958"
green3 = "#91D68D" #0AA16C 8ED18A 91D68D
orange = "#DB8436"  
brown  = "#613D0C"  
lila  = "#532A80"  
grey = "#525252"   

red5 = "#C96A57"
blue4 = "#8CC6FF"
green3 = "#9CFFA5"