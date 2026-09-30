# -*- coding: utf-8 -*-

"""
Created on 25. 08. 2026 at 20:59:30

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Inicializační strategie pro nehierarchické shlukování.

    Vzorový příklad (``RandomUniformInit``) ukazuje, jak subclassa používá
    ``self._rng``. Studenti implementují ``ForgyInit`` a ``KMeansPlusPlusInit``
    ve stejném stylu.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Initializer(ABC):
    """Abstraktní základ pro inicializační strategie těžišť.

    Zajišťuje jednotnou správu generátoru náhodných čísel přes getter/setter
    vlastnosti ``random_state``. Subclassy volají ``self._rng`` pro veškerou
    náhodnost — nikdy přímo ``np.random``.
    """

    def __init__(self, random_state: int | None = None) -> None:
        """Inicializuje inicializátor se zadaným zárodkem náhodnosti.

        Parameters
        ----------
        random_state:
            Zárodek generátoru (``None`` → nedeterministický).
        """
        self._random_state: int | None = random_state
        self._rng: np.random.Generator = np.random.default_rng(random_state)

    @property
    def random_state(self) -> int | None:
        """Vrátí aktuální zárodek generátoru náhodných čísel."""
        return self._random_state

    @random_state.setter
    def random_state(self, value: int | None) -> None:
        """
        Nastaví seed a znovu vytvoří generátor náhodných čísel.

        Setter pro vlastnost `random_state`. Kromě uložení nové hodnoty
        musí přeseedovat `self._rng`, jinak by se změna seedu na chování
        generátoru neprojevila.

        Parametry:
        ----------
        value : int | None
            Nový seed. None znamená nedeterministickou inicializaci.
        """
        self._random_state = value
        self._rng = np.random.default_rng(value)

    @abstractmethod
    def initialize(self, x: np.ndarray, k: int) -> np.ndarray:
        """Vybere počáteční těžiště pro ``k`` shluků.

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        k:
            Požadovaný počet shluků.

        Returns
        -------
        np.ndarray
            Počáteční těžiště, tvar ``(k, n_příznaků)``.
        """


class RandomUniformInit(Initializer):
    """Inicializace náhodným výběrem z rovnoměrného rozdělení v rozsahu dat.

    Pro každý příznak je rozsah určen minimem a maximem ve sloupcích matice ``X``.
    Těžiště nemusí odpovídat žádnému skutečnému bodu dat.

    Slouží jako referenční vzor: ukazuje, jak podtřída používá ``self._rng``.
    """

    def initialize(self, x: np.ndarray, k: int) -> np.ndarray:
        """Vygeneruje ``k`` těžišť náhodně z rovnoměrného rozdělení.

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        k:
            Počet těžišť (= počet shluků).

        Returns
        -------
        np.ndarray
            Těžiště tvaru ``(k, n_příznaků)`` — hodnoty v rozsahu sloupců ``x``.
        """
        # assert: Ověřte, že x je 2D matice, k >= 2 a k nepřekračuje počet bodů
        assert x.ndim == 2, "x musí být 2D matice"
        assert k >= 2, "k musí být alespoň 2"
        assert k <= x.shape[0], "k nesmí být větší než počet bodů"

        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)

        # Náhodná těžiště v rozsahu jednotlivých příznaků
        return self._rng.uniform(
            low=x_min,
            high=x_max,
            size=(k, x.shape[1]),
        )

class ForgyInit(Initializer):
    """Inicializace metodou Forgy — těžiště jsou náhodně vybrané existující body.

    Na rozdíl od ``RandomUniformInit`` jsou těžiště vždy body z datasetu,
    nikoli syntetické body v prostoru. Tato metoda je základem toho, co se
    ve většině implementací označuje jako „náhodná inicializace k-means".
    """

    def initialize(self, x: np.ndarray, k: int) -> np.ndarray:
        """Vybere ``k`` náhodných řádků z ``x`` jako počáteční těžiště.

        Úkol:
            Implementujte výběr ``k`` různých řádků z matice ``x`` pomocí
            ``self._rng``. Použijte ``self._rng.choice`` s ``replace=False``,
            aby žádný bod nebyl vybrán dvakrát.

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        k:
            Počet těžišť (= počet shluků).

        Returns
        -------
        np.ndarray
            Těžiště tvaru ``(k, n_příznaků)`` — podmnožina řádků ``x``.
        """
        # assert: Ověřte, že x je 2D matice, k >= 2 a k nepřekračuje počet bodů
        # (nelze vybrat více různých bodů než jich existuje)
        assert x.ndim == 2, "x musí být 2D matice"
        assert k >= 2, "k musí být alespoň 2"
        assert k <= x.shape[0], "k nesmí být větší než počet bodů"

        indices = self._rng.choice(
            x.shape[0],
            size=k,
            replace=False,
        )

        return x[indices].copy()

class KMeansPlusPlusInit(Initializer):
    """Inicializace k-means++ — chytré rozmístění počátečních těžišť.

    Snižuje pravděpodobnost konvergence do špatného lokálního minima tím,
    že každé další těžiště volí s pravděpodobností úměrnou čtvercové vzdálenosti
    k nejbližšímu již zvoleném těžišti.
    """

    def initialize(self, x: np.ndarray, k: int) -> np.ndarray:
        """Implementuje algoritmus k-means++ pro výběr počátečních těžišť.

        Úkol:
            Implementujte algoritmus k-means++ krok za krokem:

            1. Vyberte první těžiště náhodně (rovnoměrně) z bodů ``x``.
            2. Pro každý bod vypočítejte čtvercovou vzdálenost k nejbližšímu
               dosud zvolenému těžišti: ``D(x)^2``.
            3. Vyberte další těžiště s pravděpodobností úměrnou ``D(x)^2``
               (použijte ``self._rng.choice`` s parametrem ``p=...``).
            4. Opakujte kroky 2–3, dokud nemáte ``k`` těžišť.

            Nápověda: vzdálenost v kroku 2 počítejte jako euklidovskou
            (``np.linalg.norm``), ale nemusíte volat ``self.distance`` —
            inicializátor je nezávislý na injektované metrice shlukování.

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        k:
            Počet těžišť (= počet shluků).

        Returns
        -------
        np.ndarray
            Těžiště tvaru ``(k, n_příznaků)`` vybraná algoritmem k-means++.
        """
        # assert: Ověřte, že x je 2D matice, k >= 2 a k nepřekračuje počet bodů
        assert x.ndim == 2, "x musí být 2D matice"
        assert k >= 2, "k musí být alespoň 2"
        assert k <= x.shape[0], "k nesmí být větší než počet bodů"

        n_points = x.shape[0]

        first_idx = self._rng.integers(n_points)
        centroids = [x[first_idx].copy()]

        min_squared_distances = np.sum((x - centroids[0]) ** 2, axis=1)

        for _ in range(1, k):
            total = np.sum(min_squared_distances)

            if total == 0:
                next_idx = self._rng.integers(n_points)
            else:
                probabilities = min_squared_distances / total
                next_idx = self._rng.choice(n_points, p=probabilities)

            new_centroid = x[next_idx].copy()
            centroids.append(new_centroid)

            squared_distances = np.sum(
                (x - new_centroid) ** 2,
                axis=1,
            )

            min_squared_distances = np.minimum(
                min_squared_distances,
                squared_distances,
            )

        return np.array(centroids)
