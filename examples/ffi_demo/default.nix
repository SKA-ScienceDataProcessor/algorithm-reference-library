# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
#
# NiX build for ARL FFI work. Why NiX? See
# https://www.mrao.cam.ac.uk/~bn204/publications/2016/2016-10-nix-casa.pdf
# 
# Don't use unless you know NiX or have the time to learn
# 
{ system ? builtins.currentSystem , crossSystem ? null, config ? {}}:
let
  pkgs=(import <nixpkgs>) { inherit system crossSystem config; };

  cfitsio = pkgs.stdenv.mkDerivation {
  name = "cfitsio-3.21";

  src = pkgs.fetchurl {
    url = ftp://heasarc.gsfc.nasa.gov/software/fitsio/c/cfitsio3210.tar.gz;
    sha256 = "1ffr3p5dy2b1vj9j4li5zf22naavi9wcxsvqy236fc0ykfyip96i";
  };

    postInstall = ''
    ln -sv $out/include $out/include/cfitsio
  '';

  # Shared-only build
  #  buildFlags = "shared";
#  patchPhase =
#   '' sed -e '/^install:/s/libcfitsio.a //' -e 's@/bin/@@g' -i Makefile.in
#   '';

  };

  astropy = pkgs.python3Packages.buildPythonPackage rec {
     name   = "astropy-${version}";
     version= "2.0.2";
     # need to put in more buildInputs here so that it does not cobble
     # random stuff together off the net
     propagatedBuildInputs =
     [ pkgs.python3Packages.numpy
       pkgs.python3Packages.pytest ];

    doCheck = false;

     src = pkgs.fetchurl {
       url = "https://pypi.python.org/packages/3f/aa/9950a7c4bad169d903ef27b3a9caa7d55419fbe92423d30a6f4c0886bf09/astropy-2.0.2.tar.gz";
       sha256 = "0g8h9rspsr9dfl2bczyr622zv984gxi4r9svssr7jg8pn4ia8i25";
    };
  };

  reproject = pkgs.python3Packages.buildPythonPackage rec {
     name   = "reproject-${version}";
     version= "0.3.2";
     # need to put in more buildInputs here so that it does not cobble
     # random stuff together off the net
     propagatedBuildInputs =  [  pkgs.python3Packages.numpy astropy ];

    doCheck = false;

     src = pkgs.fetchurl {
       url = "https://pypi.python.org/packages/04/19/469356b8a860ed47d24eaa875cd503fdc626eea1b1b1337d3c3fae869b26/reproject-0.3.2.tar.gz";
       sha256 = "0lf5aiqg8faq49nv74wnj2gm31615c4r5qnclmnghxyb30j9y9s9";
    };
  };

  photutils = pkgs.python3Packages.buildPythonPackage rec {
     name   = "photutils-${version}";
     version= "0.4";
     propagatedBuildInputs =  [ pkgs.python3Packages.numpy
                                pkgs.python3Packages.six
                                astropy ];

    doCheck = false;
    src = pkgs.fetchurl {
        url = "https://pypi.python.org/packages/70/db/28fcc4447d64c0d2b9f7323d932eff0e6c17ba624d198a26b239a2c11983/photutils-0.4.tar.gz";
       sha256 = "0vrdwq4blkyk5fk21qx1zdm33c9f7cy407p2dsvay49z87hmkrki";
    };
  };

in pkgs.python3Packages.buildPythonPackage rec {
   name="sdp-arl-ffi";

   propagatedBuildInputs = [pkgs.ncurses pkgs.python3
   pkgs.python3Packages.numpy pkgs.python3Packages.scipy
   pkgs.python3Packages.cffi
   reproject astropy photutils
   pkgs.git-lfs
   cfitsio pkgs.gcc ];

}
