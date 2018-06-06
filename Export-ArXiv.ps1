#region Bootstrap PoShTeX
$modules = Get-Module -ListAvailable -Name posh-tex;
if (!$modules) {Install-Module posh-tex -Scope CurrentUser}
if (!($modules | ? {$_.Version -ge "0.1.5"})) {Update-Module posh-tex}
Import-Module posh-tex -Version "0.1.5"
#endregion

Export-ArXivArchive @{
    ProjectName = "nv-adatptive";
    TeXMain = "tex/nv-adaptive.tex";
    RenewCommands = @{
        "figurefolder" = "./fig";
    };
    AdditionalFiles = @{
        # TeX Stuff #
        "tex/*.sty" = "/";
        "fig/online-timing-diagram.pdf" = "fig/";
        "fig/experiment-pulse-sequences.pdf" = "fig/";
        "fig/risk-summary.pdf" = "fig/";
        "fig/tracking-example.pdf" = "fig/";
        "fig/heuristic-comparison.pdf" = "fig/";
        "fig/param-learning-rates.pdf" = "fig/";
        "fig/param-learning-rates-tight.pdf" = "fig/";
        "fig/param-posteriors.pdf" = "fig/";
        "fig/param-posteriors-tight.pdf" = "fig/";
        "fig/effective-strong-meas.pdf" = "fig/";
        "fig/full_risk_sampling_numbers.pdf" = "fig/";
    };
    Notebooks = @(
    )
} -Verbose


