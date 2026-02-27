\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{notes}[2025/09/03 Custom notes class]

% Load the base class
\LoadClass[11pt]{article}

% Load common packages
\RequirePackage[utf8]{inputenc}
\RequirePackage{amsmath, amssymb, amsthm}
\RequirePackage{mathtools}
\RequirePackage{physics}
\RequirePackage{cancel}

\RequirePackage{xcolor}
\RequirePackage{fullpage}
\RequirePackage{tcolorbox}
\RequirePackage{float}
\RequirePackage[style=alphabetic]{biblatex}
\RequirePackage{caption}
\captionsetup{
  margin=0.5cm,
  labelfont=bf,
  font=it
}

\RequirePackage{hyperref}
\RequirePackage{csquotes}
\RequirePackage[bottom]{footmisc}

\definecolor{cite}{RGB}{119, 104, 133}
\definecolor{link}{RGB}{224, 141, 121}
\hypersetup{
  colorlinks=true,
  citecolor=cite,
  linkcolor=link,
}

\RequirePackage[capitalise,nameinlink]{cleveref}

\RequirePackage[a4paper, margin=2.5cm,nohead, includeheadfoot]{geometry}

% Fonts

\RequirePackage{nunito}

% Theorems and commands

\newcommand{\ton}{\text{on}}
\newcommand{\tin}{\text{in}}