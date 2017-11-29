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

  astropy = pkgs.python3Packages.buildPythonPackage rec {
     name   = "astropy-${version}";
     version= "1.3";
     # need to put in more buildInputs here so that it does not cobble
     # random stuff together off the net
     propagatedBuildInputs =  [  pkgs.python3Packages.numpy ];

    doCheck = false;

     src = pkgs.fetchurl {
       url = "https://pypi.python.org/packages/51/88/c8c4355ff49f40cc538405e86af3395e76a8f9063a19cc5513a7c364d36d/astropy-1.3.tar.gz";
       sha256 = "0f3dkipzy1d61zvsavmlsllbkr4y8q5a7i02rpij9gia9233xpj9";
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

in pkgs.stdenv.mkDerivation rec {
   name="sdp-arl-ffi";

   buildInputs = [pkgs.ncurses pkgs.python3
   pkgs.python3Packages.numpy pkgs.python3Packages.scipy
   pkgs.python3Packages.cffi
   reproject astropy pkgs.git-lfs ];

}
