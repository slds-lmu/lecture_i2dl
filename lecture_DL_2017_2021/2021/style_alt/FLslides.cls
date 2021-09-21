%%% (C) 2007 Friedrich Leisch

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{FLslides}[2009/09/29 Friedrich Leisch]

\LoadClass[landscape]{slides}

%%%**********************************************************

\RequirePackage{ifthen}

\DeclareOption*{\PassOptionsToPackage{\CurrentOption}{texpower}}
\ProcessOptions*

%%%**********************************************************

\RequirePackage{color}

\definecolor{darkred}{rgb}{0.56, 0, 0}
\definecolor{darkgreen}{rgb}{0, 0.56, 0}
\definecolor{darkgrey}{rgb}{0.3, 0.3, 0.3}
\definecolor{darkblue}{rgb}{0, 0, 0.56}

\definecolor{tublau}{rgb}{0,0.4,0.66} % ca. #0066AA
\definecolor{rotgold}{rgb}{0.8,0.3,0}
\definecolor{lmugruen}{rgb}{0,0.38,0.16}
    
\definecolor{headcolor}{rgb}{0,0.38,0.16}
\definecolor{linkcolor}{rgb}{0.56, 0.56, 0.56}

\definecolor{Sinput}{rgb}{0,0,0.56}
\definecolor{Scode}{rgb}{0,0,0.56}
\definecolor{Soutput}{rgb}{0.56,0,0}

\RequirePackage{graphicx}

%%%**********************************************************


\RequirePackage[paperwidth=300mm,paperheight=225mm,
                textheight=158mm,
                lmargin=22mm,tmargin=26mm]{geometry}
\RequirePackage{fancyhdr}

\headheight20mm
\footskip16mm

\fancyhf[]{}
\renewcommand{\headrulewidth}{0pt}


%%%**********************************************************

\IfFileExists{paralist.sty}{\RequirePackage{paralist}}{}{}
\IfFileExists{texpower.sty}{\RequirePackage{soul}
  \RequirePackage[coloremph]{texpower}

  \replacecolor{textcolor}{black}
  \replacecolor{emcolor}{darkred}
  \replacecolor{codecolor}{darkblue}
}{}{}

\RequirePackage{hyperref}

\hypersetup{%
  hyperindex,%
  colorlinks,%
  linktocpage,%
  plainpages=false,%
  linkcolor=darkblue,%
  citecolor=darkblue,%
  urlcolor=lmugruen,%
  pdfstartview=Fit%
}


%%%**********************************************************

\newcounter{SlideNr}

\newcommand{\folientitel}[1]{%
  \addtocounter{SlideNr}{1}
  \fancyhead[L]{\Large\bf\textcolor{headcolor}{#1}}
  \pdfbookmark[1]{#1}{Slide.\arabic{SlideNr}}
}

\newcommand{\nfolie}[1]{
  \newpage
  \folientitel{#1}
}

\newcommand{\newslide}[1]{\nfolie{#1}}

\fancypagestyle{section}{\fancyhf{}
    \renewcommand{\headrulewidth}{0pt}
}

\setcounter{tocdepth}{2}
\newcounter{SectionNr}
\newcommand{\section}[2][]{
  \newpage
  \thispagestyle{section}
  \addtocounter{SectionNr}{1}
  \centerline{\Large\bf\textcolor{emcolor}{#2}}
  \pdfbookmark[0]{#2}{Section.\arabic{SectionNr}}
  \medskip
  \centerline{#1}
  \newpage
}

\newcommand{\subsection}[1]{
  {\large\bf #1}
  }
\newcommand{\subsubsection}[1]{
  {\textbf{#1}}
}

\newcommand{\location}[1]{\def\@location{#1}}
\location{}
\newcommand{\loczusatz}[1]{\def\@loczusatz{#1}}
\loczusatz{}
\newcommand{\affiliation}[1]{\def\@affiliation{#1}}
\affiliation{}
\newcommand{\titlefoot}[1]{\def\@tfoot{#1}}
\titlefoot{}

\newcommand{\leftfoot}[1]{\fancyfoot[L]{\tiny \textcolor{darkgrey}{#1}}}
\newcommand{\rightfoot}[1]{\fancyfoot[R]{\tiny \textcolor{darkgrey}{#1}}}
\rightfoot{\arabic{page}}

\renewcommand{\maketitle}{%
  \begin{center}
    \titlelogo

    \Large
    \bigskip
    \textbf{\@title}

    \normalsize
    \bigskip
    {\bf \@author}

    \smallskip
    {\@affiliation}
    
    
    \bigskip
    {\small\it \@location\@loczusatz}
    
    \bigskip
    {\tiny \@tfoot}
  \end{center}
  \newpage
  \renewcommand{\headrulewidth}{0.4pt}
  \pagestyle{fancy}
}

%%%**********************************************************

\newcommand{\fig}[1]{
  \begin{center}
    \includegraphics{#1}
  \end{center}
}

\newcommand{\hfig}[2][0.9]{
  \begin{center}
    \includegraphics[height=#1\textheight]{#2}
  \end{center}
}

\newcommand{\wfig}[2][0.9]{
  \begin{center}
    \includegraphics[width=#1\textwidth]{#2}
  \end{center}
}

\newcommand{\twofig}[2]{
  \includegraphics[width=.48\textwidth]{#1}
  \hfill
  \includegraphics[width=.48\textwidth]{#2}
}

\newcommand{\tafel}[1]{\ifdisplay\vfill\textcolor{darkblue}{$\to$ {#1}}\fi}

%%%**********************************************************

%%% Beamer Compat

\RequirePackage{verbatim}

\newboolean{fls@beamer}
\setboolean{fls@beamer}{false}
\ifthenelse{\boolean{display}}{%% bei display wird alles ausgegeben
  \newenvironment{folie}[1][]{\newpage}{}
}{%% bei printout werden beamer/display frames nicht ausgegeben
  \newenvironment{folie}[1][printout]{%
    \ifthenelse{\equal{#1}{beamer}}{\setboolean{fls@beamer}{true}}{}
    \ifthenelse{\boolean{fls@beamer}}{%
      \comment
    }{}
    \newpage
  }{%
    \ifthenelse{\boolean{fls@beamer}}{%
      \endcomment
      \setboolean{fls@beamer}{false}}{}
  }
}

%%%**********************************************************

\IfFileExists{fancyvrb.sty}{\RequirePackage{fancyvrb}{}{}
  
  \newenvironment{Schunk}{\par\begin{minipage}{\textwidth}}{\end{minipage}}

  \DefineVerbatimEnvironment{Sinput}{Verbatim}
  {formatcom={\color{Sinput}},fontsize=\small}
  \DefineVerbatimEnvironment{Soutput}{Verbatim}
  {formatcom={\color{Soutput}},fontsize=\small}
  \DefineVerbatimEnvironment{Scode}{Verbatim}
  {formatcom={\color{Scode}},fontsize=\small}
}

%%%**********************************************************

\hyphenpenalty=5000
\tolerance=1000

\author{Friedrich Leisch}
\affiliation{Institut f{\"u}r Statistik\\
  Ludwig-Maximilians-Universit{\"a}t M{\"u}nchen}


%%%**********************************************************

\headsep10mm

\RequirePackage{wallpaper}

\renewcommand{\maketitle}{%
  \ThisCenterWallPaper{1}{LMU-hintergrund1.png}
    
  \Large
  \vspace*{1mm}
  \textbf{\@title}

  \normalsize
  \bigskip
  {\bf \@author}

  \smallskip
  {\@affiliation}
    
    
  \bigskip
  {\small\it \@location\@loczusatz}
    
  \bigskip
  {\tiny \@tfoot}

  \newpage
      
  \renewcommand{\headrulewidth}{0.4pt}
  \pagestyle{fancy}
    
  \replacecolor{emcolor}{darkred}
  \replacecolor{headcolor}{lmugruen}
}
