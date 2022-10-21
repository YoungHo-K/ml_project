

def set_default_figure(ax):
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.1)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1.1)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1.1)
    ax.grid(b=True, axis='y', color='black', alpha=0.1, linestyle='--')

    return ax

