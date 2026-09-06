% ============================================================
% indojc.cls -- manuscript class for the Indonesian Journal on
% Computing (Indo-JC), Telkom University.
%
% Reconstructed from the official Indo-JC LaTeX template
% (Overleaf "indojc-template") and the journal's author guide:
%   - US Letter paper, 30 mm margins, single column
%   - Times-family fonts, body 10 pt
%   - Title 24 pt / authors 11 pt / affiliation 10 pt /
%     email & abstract 9 pt / captions 8 pt
%   - Level-1 headings: centered, small caps, Roman numerals
%   - Level-2 headings: italic, left-justified, capital letters
%   - Level-3 headings: run-in italic, "N)" + colon
%   - Figure captions below (8 pt, "Fig. N."); table captions
%     above (8 pt small caps, "TABLE N")
%   - IEEE-style numeric references
% ============================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{indojc}[2026/08/30 v1.0 Indo-JC manuscript class]

\LoadClass[10pt]{article}

% --- indent the first paragraph after every heading (the official
%     Indo-JC template indents all paragraphs; LaTeX's article class
%     suppresses the indent after sectioning commands by default) ---
\RequirePackage{indentfirst}

% --- geometry: US Letter, 30 mm margins, one column ---
\RequirePackage[letterpaper,left=30mm,right=30mm,top=30mm,bottom=30mm]{geometry}

% --- packages used by the manuscript (amsmath/amssymb must come
%     before newtxmath so \Bbbk and friends are not redefined) ---
\RequirePackage{amsmath,amssymb}
\RequirePackage{newtxtext}
\RequirePackage{newtxmath}

\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{booktabs}
\RequirePackage{multirow}
\RequirePackage{lettrine}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage[hyphens]{url}
\RequirePackage[pageanchor=false]{hyperref}
\RequirePackage{titlesec}

\hypersetup{hidelinks}

% --- heading numbering ---
\renewcommand{\thesection}{\Roman{section}}
\renewcommand{\thesubsection}{\Alph{subsection}}
\renewcommand{\thesubsubsection}{\arabic{subsubsection}}

% --- headings: level-1 centered small caps; level-2 italic;
%     level-3 run-in italic, "N)" label and trailing colon ---
\titleformat{\section}
  {\normalsize\centering\scshape}
  {\Roman{section}.}
  {0.5em}
  {}
\titleformat{\subsection}
  {\normalsize\itshape}
  {\Alph{subsection}.}
  {0.5em}
  {}
\titleformat{\subsubsection}[runin]
  {\normalsize\itshape}
  {\arabic{subsubsection})}
  {0.5em}
  {}
  [:]

\titlespacing*{\section}{0pt}{18pt}{12pt}
\titlespacing*{\subsection}{0pt}{12pt}{6pt}
\titlespacing*{\subsubsection}{0pt}{9pt}{5pt}

% --- figure/table numbering and captions ---
\renewcommand{\thetable}{\Roman{table}}

\captionsetup[figure]{font=footnotesize,labelsep=period}
\captionsetup[table]{font=footnotesize,labelfont=sc,labelsep=newline,
  textfont=sc,position=top}

% --- title block ---
\newcommand{\jtitle}[1]{%
  \begin{center}{\fontsize{24}{30}\selectfont\bfseries #1}\end{center}
  \par\vspace{8pt}}
\newcommand{\jauthors}[1]{%
  \begin{center}{\fontsize{11}{15}\selectfont #1}\end{center}
  \par\vspace{4pt}}
\newcommand{\jaffil}[1]{%
  \begin{center}{\fontsize{10}{13}\selectfont #1}\end{center}
  \par\vspace{2pt}}
\newcommand{\jemail}[1]{%
  \begin{center}{\fontsize{9}{12}\selectfont #1}\end{center}
  \par\vspace{10pt}}

% --- abstract and keywords (English) ---
\newenvironment{jabstract}{%
  \par\nobreak
  \begin{center}{\fontsize{9}{11}\selectfont\bfseries Abstract}\end{center}%
  \begingroup\fontsize{9}{11}\selectfont
  \noindent\ignorespaces
}{%
  \par\endgroup}
\newcommand{\jkeywords}[1]{%
  \par\vspace{2pt}%
  \begingroup\fontsize{9}{11}\selectfont
  \noindent\textbf{Keywords:} #1\par
  \endgroup\vspace{6pt}}

% --- abstract and keywords (Indonesian) ---
\newenvironment{jabstrak}{%
  \par\nobreak
  \begin{center}{\fontsize{9}{11}\selectfont\bfseries Abstrak}\end{center}%
  \begingroup\fontsize{9}{11}\selectfont
  \noindent\ignorespaces
}{%
  \par\endgroup}
\newcommand{\jkata}[1]{%
  \par\vspace{2pt}%
  \begingroup\fontsize{9}{11}\selectfont
  \noindent\textbf{Kata Kunci:} #1\par
  \endgroup\vspace{6pt}}

\endinput
